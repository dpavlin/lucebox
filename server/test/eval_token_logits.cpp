#include "deepseek4/deepseek4_backend.h"
#include "server/tokenizer.h"
#include "common/daemon_io.h"
#include "server/chat_template.h"
#include <cstdio>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

using namespace dflash;
using namespace dflash::deepseek4;
using namespace dflash::server;

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::printf("Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];

    std::printf("[probe] Loading tokenizer from %s\n", model_path);
    Tokenizer tokenizer;
    if (!tokenizer.load(model_path)) {
        std::printf("[probe] Failed to load tokenizer\n");
        return 2;
    }

    std::printf("[probe] Initializing DeepSeek4Backend\n");
    DeepSeek4Config cfg;
    cfg.target_path = model_path;
    cfg.device.gpu = 0;
    cfg.max_ctx = 4096;
    cfg.prefill_mode = PrefillAttentionMode::Exact;

    DeepSeek4Backend backend(cfg);
    if (!backend.init()) {
        std::printf("[probe] Failed to init backend\n");
        return 3;
    }

    std::vector<std::string> test_prompts = {
        "Repeat after me: rfid501.go",
        "Repeat after me: /home/dpavlin/koha-rfid-go/internal/rfid/rfid501.go"
    };

    for (const auto & prompt : test_prompts) {
        std::printf("\n=======================================================\n");
        std::printf("PROMPT: %s\n", prompt.c_str());
        std::printf("=======================================================\n");

        std::vector<int32_t> prompt_tokens;
        tokenizer.encode(prompt, prompt_tokens, false);

        std::printf("Prompt tokens (%zu): [", prompt_tokens.size());
        for (size_t i = 0; i < prompt_tokens.size(); i++) {
            std::printf("%d%s", prompt_tokens[i], i + 1 < prompt_tokens.size() ? ", " : "");
        }
        std::printf("]\n");

        GenerateRequest req;
        req.prompt = prompt_tokens;
        req.n_gen = 25;
        req.sampler.temp = 0.0f;

        std::vector<int32_t> out_tokens;
        NullDaemonIO io;
        GenerateResult res = backend.generate(req, io);
        if (!res.ok()) {
            std::printf("Generate failed: %s\n", res.error.c_str());
            continue;
        }

        std::string out_text = tokenizer.decode(res.tokens);
        std::printf("Output (%zu tokens): %s\n", res.tokens.size(), out_text.c_str());
        std::printf("Token sequence: [");
        for (size_t i = 0; i < res.tokens.size(); i++) {
            std::string t_str = tokenizer.decode({res.tokens[i]});
            std::printf("%d ('%s')%s", res.tokens[i], t_str.c_str(), i + 1 < res.tokens.size() ? ", " : "");
        }
        std::printf("]\n");
    }

    return 0;
}
