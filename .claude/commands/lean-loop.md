# Autonomous Work Loop

Work through plan items autonomously within safety boundaries.
Each iteration: pick → work → verify → commit → log → next.

## Arguments

`$ARGUMENTS` may contain:
- `--max-iterations N` — override project default (default: 5)
- `--scope autonomous|notify` — autonomy ceiling (default: from project.yml)
- `--sandbox none|openshell` — sandbox tier (default: from project.yml)

## Step 1: Pre-flight checks

### 1a. Load configuration

Read `.lean/project.yml`. Extract:
- `loop.max_iterations` (default 5, overridden by `--max-iterations`)
- `loop.scope` (default `autonomous`, overridden by `--scope`)
- `loop.sandbox` (default `none`, overridden by `--sandbox`)
- `loop.git.branch_prefix` (default `loop`)
- `loop.git.tag_runs` (default `true`)
- `loop.git.push_after_each` (default `false`)
- `loop.git.commit_style` (default `conventional`)
- `verification.quick` and `verification.full`
- `autonomy` policy (all four tiers)

If `loop` section is missing, use all defaults.

### 1b. Validate readiness

Check each condition and report status:

```
Loop readiness:
  [✓] project.yml exists with verification commands
  [✓] Autonomy policy defined
  [✓] Plan has N unblocked items in <scope> scope
  [✗] Git working tree is dirty (3 uncommitted files)
  [—] OpenShell not configured (sandbox: none)
```

**Hard blockers** (abort if any fail):
- `.lean/project.yml` missing or no `verification` section
- `.lean/plan.yml` missing
- Git working tree is dirty (`git status --porcelain` is non-empty)

**Soft warnings** (report but continue):
- No items have `acceptance` criteria (weakens completion gate)
- `loop.sandbox` is `openshell` but `openshell` CLI not found on PATH
  → fall back to `none` with warning

### 1c. Check stop signal

If `.lean/loop-stop` exists, print its contents and abort:
"A loop stop was requested: <contents>. Remove the file to allow
`/lean-loop` to run again."

### 1d. Identify candidates

Read `.lean/plan.yml` and filter items:
1. `status: planned` only
2. All `depends_on` IDs have `status: done`
3. Item scope falls within the allowed autonomy tier (`--scope`):
   - `autonomous` scope: only items whose work is refactoring, tests,
     docs, linting, formatting, or minor dependency patches
   - `notify` scope: also includes internal API changes, config changes,
     new utility functions

If zero candidates, report and stop.

### 1e. Present loop plan and confirm

```
Loop plan:
  Max iterations: 5
  Scope: autonomous
  Sandbox: none
  Branch: loop/2026-06-15
  Candidates (in order):
    1. ITEM-42: Add unit tests for parser module
    2. ITEM-43: Fix typo in README
    3. ITEM-45: Remove dead code in utils/

Proceed? [Y/n]
```

Wait for confirmation before starting.

## Step 2: Set up git branch

Create a dedicated loop branch from the current HEAD:

```bash
git checkout -b <branch_prefix>/<today's date>
```

For example: `loop/2026-06-15`. If the branch already exists (re-running
the loop on the same day), append a sequence number: `loop/2026-06-15-2`.

All loop commits land on this branch. The main branch stays untouched.

## Step 3: Set up sandbox (if configured)

### If sandbox is `openshell`:

1. Verify the gateway is reachable:
   ```bash
   openshell sandbox list
   ```
   If this fails, fall back to `none` with a warning.

2. Create a sandbox with the project bind-mounted:
   ```bash
   openshell sandbox create \
     --driver-config-json '{"podman":{"mounts":[{"type":"bind","source":"<project-root>","target":"/workspace","read_only":false}]}}' \
     -- bash
   ```

3. Apply the project policy if `.lean/openshell-policy.yaml` exists:
   ```bash
   openshell policy set <sandbox-name> --policy .lean/openshell-policy.yaml --wait
   ```

4. For the rest of the loop, execute all build/test/verification commands
   inside the sandbox via:
   ```bash
   openshell sandbox connect <sandbox-name> -- <command>
   ```

### If sandbox is `none`:

No setup needed. Commands execute directly.

## Step 4: Iteration loop

Track: `iteration = 0`, `completed_items = []`.

For each iteration (while `iteration < max_iterations`):

### 4a. Check exit conditions

1. Check `.lean/loop-stop` — if exists, go to Step 5
2. Re-read `.lean/plan.yml` to get fresh state
3. Filter for unblocked `planned` items within scope (same as Step 1d)
4. If no candidates remain, go to Step 5

### 4b. Select and start item

Pick the highest-priority unblocked candidate (priority first, then
lowest ID). Read the full item file at `.lean/items/<id>.yml` for
description, acceptance criteria, scope, and verification commands.

Update `.lean/plan.yml`: set item `status: in_progress`, `progress: 0`.

### 4c. Work on the item

Implement the work described in the item's `description`. Follow the
item's `scope` to know which files/directories to modify. If the item
has `acceptance` criteria, work toward satisfying all of them.

### 4d. Verify

Run verification commands in order. If sandbox is `openshell`, run
these inside the sandbox.

1. Item's own `verification.quick` (if defined in the item file)
2. Project's `verification.quick` from `.lean/project.yml`
3. Item's own `verification.full` (if defined)
4. Project's `verification.full`

**If any verification step fails: STOP the entire loop.** Do not skip
to the next item. Do not attempt to fix the failure. Leave the item
as `in_progress` so the human can investigate. Go to Step 5.

### 4e. Check acceptance criteria

If the item has `acceptance` criteria in its item file, evaluate each:
- For each criterion, verify the implementation meets it
- If any criterion is NOT MET, stop the loop (go to Step 5)

If the item has no acceptance criteria, skip this step.

### 4f. Commit

1. Run `git diff --stat` to review what changed. If any files outside the
   item's `scope` appear, exclude them from the commit (`git add` only
   the relevant files instead of `git add -A`). If files within scope
   appear that look like artifacts (`.pyc`, `.o`, editor temp files),
   exclude them too.

2. Stage the relevant changed files plus `.lean/plan.yml` and
   `.lean/activity.log`.

3. Commit with a structured message. If `commit_style` is `conventional`:

   ```
   <type>(<scope>): <item title>

   Item: <item-id>
   Acceptance:
   - [x] criterion 1
   - [x] criterion 2

   Loop iteration <N>/<max>.

   Signed-off-by: <from git config>
   ```

   Derive `<type>` from the item's nature:
   - Tests, test coverage → `test`
   - Documentation → `docs`
   - Linting, formatting → `style`
   - Dead code removal, refactoring → `refactor`
   - Bug fixes → `fix`
   - New utility functions, features → `feat`
   - Dependency changes → `build`
   - Everything else → `chore`

   If `commit_style` is `plain`, use: `<item-id>: <item title>` as the
   subject line, with the same body.

4. If `loop.git.push_after_each` is `true`:
   ```bash
   git push -u origin <branch-name>
   ```

### 4g. Update state and log

1. Update `.lean/plan.yml`: set item `status: done`, `progress: 100`
2. Append to `.lean/activity.log`:
   ```yaml
   - timestamp: "<ISO8601>"
     actor: claude
     action: loop-iteration
     item: "<item-id>"
     summary: "<one-line summary of what was done>"
     iteration: <N>
     max_iterations: <max>
   ```

3. Add item ID to `completed_items` list, increment `iteration`.

## Step 5: Loop exit

### 5a. Determine exit reason

- `.lean/loop-stop` found → "Human stop signal"
- Max iterations reached → "Reached iteration limit (N/N)"
- No candidates → "No unblocked items remaining in <scope> scope"
- Verification failed → "Verification failed on item <id>"
- Acceptance not met → "Acceptance criteria not met on item <id>"

### 5b. Tag the run

If `loop.git.tag_runs` is `true` and at least one item was completed:

```bash
git tag "loop-run/<date>-<iteration-count>"
```

For example: `loop-run/2026-06-15-3` (3 iterations completed).

### 5c. Clean up sandbox

If sandbox is `openshell`, destroy the sandbox:
```bash
openshell sandbox list  # find the sandbox name
# only destroy if it was created by this loop run
```

### 5d. Write loop summary to activity log

Append a summary entry:

```yaml
- timestamp: "<ISO8601>"
  actor: claude
  action: loop-complete
  summary: "Loop finished: <N> items completed (<ids>). Exit: <reason>"
  details:
    iterations: <N>
    max_iterations: <max>
    completed: [<item-ids>]
    exit_reason: "<reason>"
    branch: "<branch-name>"
    tag: "<tag-name or null>"
```

### 5e. Report

```
Loop complete:
  Branch: loop/2026-06-15
  Tag: loop-run/2026-06-15-3
  Iterations: 3 of 5
  Completed:
    - ITEM-42: Add unit tests for parser module (test)
    - ITEM-43: Fix typo in README (docs)
    - ITEM-45: Remove dead code in utils/ (refactor)
  Stopped: No unblocked items remaining in autonomous scope
  Skipped (needs approval):
    - ITEM-44: Add new CLI subcommand (approve scope)
  Next steps:
    - Review the loop branch: git log loop/2026-06-15
    - Merge when satisfied: git merge loop/2026-06-15
```
