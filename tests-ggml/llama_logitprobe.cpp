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

#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s model.gguf tokens.txt logits.bin [all]\n", argv[0]);
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
    cp.n_ctx   = 4096;
    cp.n_batch = 4096;
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

    const int n = (int) tokens.size();
    llama_batch batch = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; i++) {
        llama_batch_add(batch, tokens[i], i, {0}, true);
    }

    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "decode failed\n");
        return 1;
    }

    const int nv = llama_n_vocab(model);
    FILE * out = fopen(argv[3], "wb");
    if (!out) {
        fprintf(stderr, "failed to open output file\n");
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
    fclose(out);

    fprintf(stderr, "wrote %s: %d token(s), vocab %d%s\n", argv[3], n, nv,
            dump_all ? " (all positions)" : " (last position)");

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
