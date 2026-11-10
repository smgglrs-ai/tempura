// llama_logitprobe -- reference logits from llama.cpp for an exact token-id
// sequence. Used by the tempura quality-bisect workflow: tempura's
// bench-perplexity TEMPURA_PPL_PROBE dumps its own logits for the same ids,
// and the two are compared position by position.
//
// Why a C tool and not a reimplementation: llama.cpp is the project's
// designated ground truth. This links llama.cpp's own model code and dumps
// its output -- no re-derived math involved.
//
// Build (on the GPU host, against the local llama.cpp checkout):
//   g++ -O2 -std=c++17 -I ~/Code/github.com/llama.cpp/include \
//     -I ~/Code/github.com/llama.cpp/ggml/include \
//     llama_logitprobe.cpp -L ~/Code/github.com/llama.cpp/build/bin -lllama \
//     -o /tmp/llama-logitprobe
// Run:
//   LD_LIBRARY_PATH=~/Code/github.com/llama.cpp/build/bin /tmp/llama-logitprobe \
//     model.gguf tokens.txt logits.bin [all]
//
//   tokens.txt  whitespace-separated llama_token ids (no BOS added here --
//               put the BOS id in the file explicitly when comparing against
//               tempura, whose tokenizer prepends it)
//   logits.bin  f32 little-endian. Default: the LAST position's logits
//               (matches InferenceModel::forward semantics). With the 4th
//               arg "all": n_tokens * n_vocab floats, position-major.
//               With the 4th arg "batch": tokens.txt is ONE PROMPT PER LINE
//               (ids whitespace-separated); logits.bin gets one last-position
//               logit vector per line, in order. The KV cache is cleared
//               between prompts. Use for many short prompts (MMLU) without
//               reloading the model per prompt.

#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    // tokenize mode: llama_logitprobe model.gguf tokenize text.txt out.ids
    // Writes llama.cpp's tokenization (add_special=true) one id per line --
    // tokenization diff against tempura's TEMPURA_PPL_DUMP_TOKENS output.
    if (argc == 5 && std::string(argv[2]) == "tokenize") {
        llama_backend_init();
        llama_model_params mp = llama_model_default_params();
        llama_model * model = llama_model_load_from_file(argv[1], mp);
        if (!model) {
            fprintf(stderr, "failed to load model\n");
            return 1;
        }
        const llama_vocab * vo = llama_model_get_vocab(model);
        FILE * tf = fopen(argv[3], "rb");
        if (!tf) {
            fprintf(stderr, "failed to open text file\n");
            return 1;
        }
        fseek(tf, 0, SEEK_END);
        long sz = ftell(tf);
        fseek(tf, 0, SEEK_SET);
        std::string text(sz, '\0');
        if (fread(text.data(), 1, sz, tf) != (size_t)sz) {
            fprintf(stderr, "failed to read text file\n");
            return 1;
        }
        fclose(tf);
        std::vector<llama_token> toks(text.size() + 16);
        int n = llama_tokenize(vo, text.c_str(), text.size(), toks.data(),
                               toks.size(), true, false);
        if (n < 0) {
            fprintf(stderr, "tokenize failed (rc %d)\n", n);
            return 1;
        }
        FILE * out = fopen(argv[4], "w");
        for (int i = 0; i < n; i++) {
            char piece[64];
            int pl = llama_token_to_piece(vo, toks[i], piece, sizeof(piece), 0, false);
            if (pl < 0) pl = 0;
            piece[pl < 63 ? pl : 63] = '\0';
            fprintf(out, "%d\t%s\n", toks[i], piece);
        }
        fclose(out);
        fprintf(stderr, "tokenized %zu bytes -> %d tokens\n", text.size(), n);
        return 0;
    }
    if (argc < 4) {
        fprintf(stderr,
                "usage: %s model.gguf tokens.txt logits.bin [all|batch]\n"
                "       %s model.gguf tokenize text.txt out.ids\n",
                argv[0], argv[0]);
        return 1;
    }
    const bool dump_all = argc > 4 && std::string(argv[4]) == "all";

    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) {
        fprintf(stderr, "failed to load model\n");
        return 1;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx    = 4096;
    cp.n_batch  = 4096;
    cp.n_ubatch = 4096;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "failed to create context\n");
        return 1;
    }

    FILE * tf = fopen(argv[2], "r");
    if (!tf) {
        fprintf(stderr, "failed to open token file\n");
        return 1;
    }
    const bool batch_mode = argc > 4 && std::string(argv[4]) == "batch";
    std::vector<std::vector<llama_token>> prompts;
    if (batch_mode) {
        // One prompt per line: whitespace-separated token ids.
        static char line[1 << 20];
        while (fgets(line, sizeof(line), tf)) {
            std::vector<llama_token> toks;
            int t, used = 0;
            const char * p = line;
            while (sscanf(p, "%d%n", &t, &used) == 1) {
                toks.push_back(t);
                p += used;
            }
            if (!toks.empty()) {
                prompts.push_back(std::move(toks));
            }
        }
        fclose(tf);
        if (prompts.empty()) {
            fprintf(stderr, "no prompts in file\n");
            return 1;
        }
    } else {
        std::vector<llama_token> tokens;
        int t;
        while (fscanf(tf, "%d", &t) == 1) {
            tokens.push_back(t);
        }
        fclose(tf);
        if (tokens.empty()) {
            fprintf(stderr, "no tokens in file\n");
            return 1;
        }
        prompts.push_back(std::move(tokens));
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int nv = llama_vocab_n_tokens(vocab);
    FILE * out = fopen(argv[3], "wb");
    if (!out) {
        fprintf(stderr, "failed to open output file\n");
        return 1;
    }
    int n_total = 0;
    for (const auto & tokens : prompts) {
        const int n = (int) tokens.size();
        llama_batch_ext * batch = llama_batch_ext_init(ctx);
        llama_batch_ext_clear(batch);
        for (int i = 0; i < n; i++) {
            const int32_t idx = llama_batch_ext_add_token(batch, /*seq_id*/ 0, tokens[i]);
            if (idx < 0) {
                fprintf(stderr, "batch add failed at %d\n", i);
                return 1;
            }
            llama_pos pos = i;
            llama_batch_ext_set_pos(batch, idx, &pos);
            llama_batch_ext_set_output_logits(batch, idx, true);
        }

        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch) != 0) {
            fprintf(stderr, "decode failed\n");
            return 1;
        }

        if (dump_all) {
            for (int i = 0; i < n; i++) {
                const float * l = llama_get_logits_ith(ctx, i);
                fwrite(l, sizeof(float), nv, out);
            }
        } else {
            const float * l = llama_get_logits_ith(ctx, n - 1);
            fwrite(l, sizeof(float), nv, out);
        }
        llama_batch_ext_free(batch);
        // Fresh KV per prompt (perplexity.cpp semantics).
        llama_memory_clear(llama_get_memory(ctx), true);
        n_total++;
    }
    fclose(out);

    fprintf(stderr, "wrote %s: %d prompt(s), vocab %d, last-position logits%s\n",
            argv[3], n_total, nv, dump_all ? " (all positions)" : "");

    // Optional raw state dump (env LLAMA_STATE_OUT, single-prompt mode):
    // the sequence state blob from llama_state_seq_get_data -- contains the
    // serialized KV cache (per-cell meta, then per-layer K/V tensor bytes).
    // Used for byte-level K-cache diffs against tempura.
    if (!batch_mode && !dump_all) {
        if (const char * state_out = getenv("LLAMA_STATE_OUT")) {
            const size_t state_size = llama_state_seq_get_size(ctx, 0);
            std::vector<uint8_t> state(state_size);
            const size_t written = llama_state_seq_get_data(ctx, state.data(), state.size(), 0);
            FILE * sf = fopen(state_out, "wb");
            if (sf) {
                fwrite(state.data(), 1, written, sf);
                fclose(sf);
                fprintf(stderr, "wrote %s: %zu bytes of sequence state\n", state_out, written);
            } else {
                fprintf(stderr, "failed to open %s\n", state_out);
            }
        }
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
