#include "deepseek4/deepseek4_backend.h"
#include "server/tokenizer.h"
#include "common/model_backend.h"
#include <cstdio>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

using namespace dflash::common;

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];

    std::fprintf(stderr, "[probe] Loading tokenizer from %s\n", model_path);
    Tokenizer tokenizer;
    if (!tokenizer.load_from_gguf(model_path)) {
        std::fprintf(stderr, "[probe] Failed to load tokenizer\n");
        return 2;
    }

    std::vector<std::string> patterns = {
        "rfid501.go", "rfid5o1.go",
        "rfid502.go", "rfid5o2.go",
        "rfid510.go", "rfid5l0.go",
        "item501.go", "item5o1.go",
        "501", "5o1", "502", "5o2", "510", "5l0"
    };

    std::printf("\n=======================================================\n");
    std::printf("TOKEN ENCODING COMPARISON TABLE\n");
    std::printf("=======================================================\n");
    for (const auto & p : patterns) {
        std::vector<int32_t> ids = tokenizer.encode(p);
        std::printf("%-15s -> tokens (%zu): [", p.c_str(), ids.size());
        for (size_t i = 0; i < ids.size(); i++) {
            std::string s = tokenizer.decode({ids[i]});
            std::printf("%d ('%s')%s", ids[i], s.c_str(), i + 1 < ids.size() ? ", " : "");
        }
        std::printf("]\n");
    }
    std::printf("=======================================================\n\n");
    std::fflush(stdout);

    std::fprintf(stderr, "[probe] Initializing DeepSeek4Backend\n");
    DeepSeek4BackendConfig cfg;
    cfg.model_path = model_path;
    cfg.device.gpu = 0;
    cfg.max_ctx = 4096;
    cfg.prefill_mode = PrefillAttentionMode::Exact;

    DeepSeek4Backend backend(cfg);
    if (!backend.init()) {
        std::fprintf(stderr, "[probe] Failed to init backend\n");
        return 3;
    }

    struct TestCase {
        std::string name;
        std::string prompt;
        float temp = 0.0f;
        float rep_penalty = 1.0f;
    };

    std::vector<TestCase> test_cases = {
        {"llama-cli Raw Prompt (No Chat Template)", "Repeat after me: /home/dpavlin/koha-rfid-go/internal/rfid/rfid501.go", 0.0f, 1.0f},
        {"lucebox Chat Template (<｜User｜>...<｜Assistant｜><think>)", "<｜User｜>Repeat after me: /home/dpavlin/koha-rfid-go/internal/rfid/rfid501.go<｜Assistant｜><think>", 0.0f, 1.0f}
    };

    for (const auto & tc : test_cases) {
        std::printf("\n=======================================================\n");
        std::printf("TEST CASE: %s (temp=%.2f, rep_penalty=%.2f)\n", tc.name.c_str(), tc.temp, tc.rep_penalty);
        std::printf("PROMPT: %s\n", tc.prompt.c_str());
        std::printf("=======================================================\n");

        std::vector<int32_t> prompt_tokens = tokenizer.encode(tc.prompt);

        std::printf("Prompt tokens (%zu): [", prompt_tokens.size());
        for (size_t i = 0; i < prompt_tokens.size(); i++) {
            std::printf("%d%s", prompt_tokens[i], i + 1 < prompt_tokens.size() ? ", " : "");
        }
        std::printf("]\n");

        GenerateRequest req;
        req.prompt = prompt_tokens;
        req.n_gen = 300;
        req.sampler.temp = tc.temp;
        DaemonIO io;
        GenerateResult res = backend.generate(req, io);
        if (!res.ok()) {
            std::printf("Generate failed: %s\n", std::string(res.error_detail()).c_str());
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
