#!/bin/bash
# Two-machine workflow: edit/verify/commit on the laptop, build+benchmark on
# the GPU host (fishtank.home.local, RTX 5090) over passwordless SSH.
#
# Source code moves between the machines ONLY through git origin -- never
# rsync target/, .oxide-artifacts/ or model files.
#
# Commands:
#   status                   pre-flight: GPU/RAM/instances/tmux/git both ends
#   models                   list GGUF models in the remote cache
#   sync [--wip]             push this repo, pull on fishtank (same branch)
#                               --wip: snapshot uncommitted work (tracked +
#                               untracked, .gitignore honored) onto the
#                               branch wip/<hostname> without local commits
#   bench [--no-sync] BIN [args...]
#                            sync, then build+run tools/bench-gpu BIN on
#                            fishtank inside tmux. A bare arg matching a
#                            filename in the remote model cache is rewritten
#                            to its full path. PTX is re-copied after build.
#   run 'shell command'      run an arbitrary command on fishtank in tmux
#   oxide                    cargo oxide build all GEMV/attn crates on
#                            fishtank + copy PTX into build dirs
#   watch                    attach to the remote tmux session
#   log [N]                  tail newest remote run log (default 30 lines)
#   results                  pull bench-*.json + newest log into this repo
#   stop                     kill the remote tmux session
#
# Environment overrides:
#   FISHTANK_HOST (fishtank.home.local)   FISHTANK_SESSION (fishtank)
#   FISHTANK_GPU_BUSY_MB (1500)           FISHTANK_GPU_BUSY_UTIL (10)
#   FISHTANK_ALLOW_PROCS (unset)          skip the no-other-instances check
#                                         (e.g. known idle CPU-mode servers)

set -euo pipefail

SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

HOST="${FISHTANK_HOST:-fishtank.home.local}"
SESSION="${FISHTANK_SESSION:-fishtank}"
GPU_BUSY_MB="${FISHTANK_GPU_BUSY_MB:-1500}"
GPU_BUSY_UTIL="${FISHTANK_GPU_BUSY_UTIL:-10}"

# Single-quoted on purpose: $HOME must expand on the remote host.
REMOTE_REPO='$HOME/Code/github.com/smgglrs-ai/tempura'
REMOTE_MODELS='$HOME/.cache/tempura/models'
REMOTE_RUNDIR='$HOME/.cache/tempura/fishtank'

SSH="ssh -o BatchMode=yes"

die() { echo "error: $*" >&2; exit 1; }

usage() { tail -n +2 "$SCRIPT_PATH" | awk '/^#/{sub(/^# ?/, ""); print; next} {exit}'; }

remote_ok() { $SSH "$HOST" "test -d $REMOTE_REPO"; }

# Enforce the AGENTS.md rule: never run multiple instances at the same time.
preflight() {
    remote_ok || die "no clone at ~/Code/github.com/smgglrs-ai/tempura on $HOST"
    local gpu mem util
    gpu=$($SSH "$HOST" "nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | head -1") \
        || die "cannot query the GPU on $HOST"
    mem=$(echo "$gpu" | cut -d, -f1 | tr -d ' ')
    util=$(echo "$gpu" | cut -d, -f2 | tr -d ' ')
    if [ "$mem" -gt "$GPU_BUSY_MB" ]; then
        die "GPU on $HOST is busy (${mem} MiB used > ${GPU_BUSY_MB} MiB) -- never run multiple instances (AGENTS.md)"
    fi
    if [ "$util" -gt "$GPU_BUSY_UTIL" ]; then
        die "GPU on $HOST is busy (${util}% utilization)"
    fi
    if [ -z "${FISHTANK_ALLOW_PROCS:-}" ] && $SSH "$HOST" "pgrep -f '[t]empura|[b]ench-' >/dev/null"; then
        echo "tempura/bench processes already running on $HOST:" >&2
        $SSH "$HOST" "pgrep -af '[t]empura|[b]ench-'" | grep -v 'bash -c' >&2 || true
        die "never run multiple instances (AGENTS.md) -- set FISHTANK_ALLOW_PROCS=1 to override (e.g. known idle CPU-mode servers)"
    fi
    if $SSH "$HOST" "tmux has-session -t $SESSION 2>/dev/null"; then
        die "tmux session '$SESSION' already exists on $HOST -- use watch/log/results, or stop it first"
    fi
}

# Write the runner script to the remote rundir, launch it in tmux.
launch() { # $1 = name (bench|run), $2 = stamp, $3 = script body
    local name="$1" stamp="$2" script="$3"
    $SSH "$HOST" "mkdir -p $REMOTE_RUNDIR && cat > $REMOTE_RUNDIR/$name-$stamp.sh" <<<"$script"
    $SSH "$HOST" "tmux new-session -d -s $SESSION 'bash $REMOTE_RUNDIR/$name-$stamp.sh'"
}

cmd_status() {
    echo "=== $HOST ==="
    $SSH "$HOST" '
echo "GPU:    $(nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null)"
echo "RAM:    $(free -h | grep "^Mem:")"
p=$(pgrep -af "[t]empura|[b]ench-" | grep -v "bash -c" | tr "\n" " ")
echo "procs:  ${p:-none}"
t=$(tmux ls 2>/dev/null | tr "\n" " ")
echo "tmux:   ${t:-none}"
if cd "$HOME/Code/github.com/smgglrs-ai/tempura" 2>/dev/null; then
    echo "git:    $(git status -sb | head -1)  $(git log --oneline -1)"
else
    echo "git:    no clone"
fi
log=$(ls -t "$HOME/.cache/tempura/fishtank"/*.log 2>/dev/null | head -1)
if [ -n "$log" ]; then
    echo "log:    $log"
    tail -n 3 "$log" | sed "s/^/          /"
fi'
    echo
    echo "=== laptop ==="
    git status -sb | head -1
    git log --oneline -1
}

cmd_models() {
    $SSH "$HOST" "ls -lh $REMOTE_MODELS" || die "cannot list $REMOTE_MODELS on $HOST"
}

cmd_sync() {
    local wip=0
    if [ "${1:-}" = "--wip" ]; then wip=1; fi
    remote_ok || die "no clone at ~/Code/github.com/smgglrs-ai/tempura on $HOST"
    local branch
    branch=$(git branch --show-current)
    [ -n "$branch" ] || die "detached HEAD, refusing to sync"

    if [ "$wip" = 1 ]; then
        local wip_branch="wip/$(hostname -s)"
        local tmpidx tree commit
        # WHY a throwaway index: snapshot the working tree into a branch
        # without creating local commits or touching the real index.
        tmpidx=$(mktemp)
        GIT_INDEX_FILE="$tmpidx" git read-tree HEAD
        GIT_INDEX_FILE="$tmpidx" git add -A
        tree=$(GIT_INDEX_FILE="$tmpidx" git write-tree)
        rm -f "$tmpidx"
        commit=$(git commit-tree "$tree" -p HEAD -m "wip: $(hostname -s) $(date -u +%FT%TZ)")
        git push --force origin "$commit:refs/heads/$wip_branch"
        $SSH "$HOST" "cd $REMOTE_REPO && git fetch origin $wip_branch && git checkout -B $wip_branch FETCH_HEAD && git log --oneline -1"
        echo "synced: wip snapshot -> $wip_branch (fishtank checked out on it)"
        return
    fi

    if [ -n "$(git status --porcelain -uno)" ]; then
        die "uncommitted changes in tracked files -- commit or stash them, or use 'sync --wip'"
    fi
    if git ls-remote --exit-code --heads origin "$branch" >/dev/null 2>&1; then
        git pull --ff-only origin "$branch"
        git push origin "$branch"
    else
        git push -u origin "$branch"
    fi
    $SSH "$HOST" "cd $REMOTE_REPO && git fetch origin && git checkout $branch && git pull --ff-only origin $branch && git log --oneline -1"
    echo "synced: $branch (laptop -> origin -> fishtank)"
}

cmd_bench() {
    local nosync=0
    if [ "${1:-}" = "--no-sync" ]; then nosync=1; shift; fi
    local bin="${1:-}"
    if [ -z "$bin" ]; then
        die "usage: fishtank.sh bench [--no-sync] <bin> [args...] -- bins: $(ls tools/bench-gpu/src/bench_*.rs | xargs -n1 basename | sed 's/^bench_//;s/\.rs$//' | tr '\n' ' ') (bench- prefix optional)"
    fi
    shift
    local bin_dashed="${bin//_/-}"
    case "$bin_dashed" in bench-*) ;; *) bin_dashed="bench-$bin_dashed" ;; esac
    [ -f "tools/bench-gpu/src/${bin_dashed//-/_}.rs" ] || die "unknown bench binary '$bin' (see tools/bench-gpu/Cargo.toml)"
    preflight
    if [ "$nosync" != 1 ]; then cmd_sync; fi

    local stamp args_q a runner
    stamp=$(date +%Y%m%d-%H%M%S)
    args_q=""
    for a in "$@"; do args_q+=" $(printf '%q' "$a")"; done
    runner=$(cat <<EOF
set -uo pipefail
{
cd $REMOTE_REPO || exit 1
models=$REMOTE_MODELS
args=()
for a in$args_q; do
    if [ -f "\$models/\$a" ]; then args+=("\$models/\$a"); else args+=("\$a"); fi
done
echo "=== fishtank bench: $bin_dashed \${args[*]:-} ==="
cargo build --manifest-path tools/bench-gpu/Cargo.toml --release --bin $bin_dashed || exit \$?
# WHY: a rebuild lands tempura-kernels in a new fingerprinted out dir; the
# oxide PTX must be copied there again or the loader silently falls back to
# cuTile and the benchmark numbers are wrong.
for d in target/release/build/tempura-kernels-*/out; do
    [ -d "\$d" ] || continue
    cp tools/oxide-gemv/oxide_gemv.ptx               "\$d/oxide_q2_gemv.ptx" 2>/dev/null || true
    cp tools/oxide-gemv-q4k/oxide_gemv_q4k.ptx       "\$d/oxide_q4k_gemv.ptx" 2>/dev/null || true
    cp tools/oxide-gemv-q6k/oxide_gemv_q6k.ptx       "\$d/oxide_q6k_gemv.ptx" 2>/dev/null || true
    cp tools/oxide-decode-attn/oxide_decode_attn.ptx "\$d/oxide_decode_attn.ptx" 2>/dev/null || true
done
cargo run --manifest-path tools/bench-gpu/Cargo.toml --release --bin $bin_dashed -- "\${args[@]}"
s=\$?
echo
echo "=== exit status: \$s ==="
exit \$s
} 2>&1 | tee $REMOTE_RUNDIR/bench-$stamp.log
EOF
)
    launch bench "$stamp" "$runner"
    echo "bench '$bin_dashed' running in tmux session '$SESSION' on $HOST"
    echo "  scripts/fishtank.sh watch     # attach (Ctrl-b d to detach)"
    echo "  scripts/fishtank.sh log       # tail the log"
    echo "  scripts/fishtank.sh results   # pull bench-*.json + log when done"
    echo "  scripts/fishtank.sh stop      # kill it"
}

cmd_run() {
    [ $# -gt 0 ] || die "usage: fishtank.sh run '<shell command>'  (runs on $HOST in tmux, cwd = the repo)"
    preflight
    local cmd="$*" stamp runner
    stamp=$(date +%Y%m%d-%H%M%S)
    runner=$(cat <<EOF
set -uo pipefail
{
cd $REMOTE_REPO || exit 1
$cmd
s=\$?
echo
echo "=== exit status: \$s ==="
exit \$s
} 2>&1 | tee $REMOTE_RUNDIR/run-$stamp.log
EOF
)
    launch run "$stamp" "$runner"
    echo "running in tmux session '$SESSION' on $HOST -- use watch/log/stop"
}

cmd_oxide() {
    remote_ok || die "no clone at ~/Code/github.com/smgglrs-ai/tempura on $HOST"
    $SSH "$HOST" bash -s <<'EOF'
set -euo pipefail
backend="$HOME/Code/github.com/NVLabs/cuda-oxide/crates/rustc-codegen-cuda/target/release/librustc_codegen_cuda.so"
if [ ! -f "$backend" ]; then
    echo "error: oxide backend .so missing on $(hostname): $backend" >&2
    echo "  build it: cd ~/Code/github.com/NVLabs/cuda-oxide/crates/rustc-codegen-cuda && cargo build --release" >&2
    exit 1
fi
export CUDA_OXIDE_BACKEND="$backend"
cd "$HOME/Code/github.com/smgglrs-ai/tempura"
for crate in oxide-gemv oxide-gemv-q4k oxide-gemv-q6k oxide-decode-attn; do
    echo "--- cargo oxide build: tools/$crate ---"
    (cd "tools/$crate" && cargo oxide build)
done
for d in target/release/build/tempura-kernels-*/out; do
    [ -d "$d" ] || continue
    cp tools/oxide-gemv/oxide_gemv.ptx               "$d/oxide_q2_gemv.ptx" 2>/dev/null || true
    cp tools/oxide-gemv-q4k/oxide_gemv_q4k.ptx       "$d/oxide_q4k_gemv.ptx" 2>/dev/null || true
    cp tools/oxide-gemv-q6k/oxide_gemv_q6k.ptx       "$d/oxide_q6k_gemv.ptx" 2>/dev/null || true
    cp tools/oxide-decode-attn/oxide_decode_attn.ptx "$d/oxide_decode_attn.ptx" 2>/dev/null || true
done
echo "PTX copied into tempura-kernels build dirs"
echo "note: for the upstream dp4a backend variant see AGENTS.md (RUSTUP_TOOLCHAIN=nightly-2026-10-11)"
EOF
}

cmd_watch() {
    exec ssh -t "$HOST" "tmux attach -t $SESSION"
}

cmd_log() {
    local n="${1:-30}"
    $SSH "$HOST" "l=\$(ls -t $REMOTE_RUNDIR/*.log 2>/dev/null | head -1); if [ -n \"\$l\" ]; then echo \"== \$l ==\"; tail -n $n \"\$l\"; else echo 'no run logs yet'; fi"
}

cmd_results() {
    remote_ok || die "no clone at ~/Code/github.com/smgglrs-ai/tempura on $HOST"
    if $SSH "$HOST" "cd $REMOTE_REPO && ls bench-*.json >/dev/null 2>&1"; then
        $SSH "$HOST" "cd $REMOTE_REPO && tar cf - bench-*.json" | tar xf -
        echo "pulled bench-*.json into $PROJECT_ROOT"
    else
        echo "no bench-*.json in the remote clone"
    fi
    local newest base
    newest=$($SSH "$HOST" "ls -t $REMOTE_RUNDIR/*.log 2>/dev/null | head -1" || true)
    if [ -n "${newest:-}" ]; then
        base=$(basename "$newest")
        $SSH "$HOST" "tar cf - -C $REMOTE_RUNDIR $base" | tar xf -
        mv -f "$base" "tempura-fishtank-$base"
        echo "log: $PROJECT_ROOT/tempura-fishtank-$base (gitignored)"
    fi
}

cmd_stop() {
    if $SSH "$HOST" "tmux kill-session -t $SESSION 2>/dev/null"; then
        echo "killed tmux session '$SESSION' on $HOST"
    else
        echo "no tmux session '$SESSION' on $HOST"
    fi
    if $SSH "$HOST" "pgrep -f '[t]empura|[b]ench-' >/dev/null"; then
        echo "warning: processes still alive on $HOST:"
        $SSH "$HOST" "pgrep -af '[t]empura|[b]ench-'" || true
        echo "if these are orphans, kill them manually: ssh $HOST 'pkill -f <pattern>'"
    fi
}

case "${1:-}" in
    status)  shift; cmd_status "$@" ;;
    models)  shift; cmd_models "$@" ;;
    sync)    shift; cmd_sync "$@" ;;
    bench)   shift; cmd_bench "$@" ;;
    run)     shift; cmd_run "$@" ;;
    oxide)   shift; cmd_oxide "$@" ;;
    watch)   shift; cmd_watch "$@" ;;
    log)     shift; cmd_log "$@" ;;
    results) shift; cmd_results "$@" ;;
    stop)    shift; cmd_stop "$@" ;;
    -h|--help|help|"") usage ;;
    *) usage; exit 1 ;;
esac
