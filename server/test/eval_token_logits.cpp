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
        // 1. Directory Stem & Path Repetition Variations
        {"Baseline (Full Path)", "<｜User｜>Repeat after me: /home/dpavlin/koha-rfid-go/internal/rfid/rfid501.go<｜Assistant｜></think>", 0.0f, 1.0f},
        {"Different Dir (device/rfid501)", "<｜User｜>Repeat after me: /home/dpavlin/koha-rfid-go/internal/device/rfid501.go<｜Assistant｜></think>", 0.0f, 1.0f},
        {"Different Stem (rfid/item501)", "<｜User｜>Repeat after me: /home/dpavlin/koha-rfid-go/internal/rfid/item501.go<｜Assistant｜></think>", 0.0f, 1.0f},
        {"Different Digit (rfid502)", "<｜User｜>Repeat after me: /home/dpavlin/koha-rfid-go/internal/rfid/rfid502.go<｜Assistant｜></think>", 0.0f, 1.0f},
        {"Double Zero (rfid500)", "<｜User｜>Repeat after me: /home/dpavlin/koha-rfid-go/internal/rfid/rfid500.go<｜Assistant｜></think>", 0.0f, 1.0f},
        {"Zero at End (rfid510)", "<｜User｜>Repeat after me: /home/dpavlin/koha-rfid-go/internal/rfid/rfid510.go<｜Assistant｜></think>", 0.0f, 1.0f},
        {"Filename Only (rfid501.go)", "<｜User｜>Repeat after me: rfid501.go<｜Assistant｜></think>", 0.0f, 1.0f},

        // 2. Quoting & Delimiters
        {"Double Quotes", "<｜User｜>Repeat after me: \"/home/dpavlin/koha-rfid-go/internal/rfid/rfid501.go\"<｜Assistant｜></think>", 0.0f, 1.0f},
        {"Backticks", "<｜User｜>Repeat after me: `/home/dpavlin/koha-rfid-go/internal/rfid/rfid501.go`<｜Assistant｜></think>", 0.0f, 1.0f},
        {"Single Quotes", "<｜User｜>Repeat after me: '/home/dpavlin/koha-rfid-go/internal/rfid/rfid501.go'<｜Assistant｜></think>", 0.0f, 1.0f},

        // 3. System Directives & Instruction Framing
        {"System Exact-Copy Directive", "<｜begin▁of▁sentence｜>You are a literal copy engine. Output exact characters and digits verbatim without alteration.<｜User｜>Repeat after me: /home/dpavlin/koha-rfid-go/internal/rfid/rfid501.go<｜Assistant｜></think>", 0.0f, 1.0f},
        {"Explicit Digit Directive", "<｜User｜>Repeat the path /home/dpavlin/koha-rfid-go/internal/rfid/rfid501.go exactly (it contains digit zero '0', not letter 'o'):<｜Assistant｜></think>", 0.0f, 1.0f},
        {"Character By Character Directive", "<｜User｜>Repeat after me character by character: /home/dpavlin/koha-rfid-go/internal/rfid/rfid501.go<｜Assistant｜></think>", 0.0f, 1.0f},

        // 4. Repetition Penalty Variations
        {"Repetition Penalty 1.05", "<｜User｜>Repeat after me: /home/dpavlin/koha-rfid-go/internal/rfid/rfid501.go<｜Assistant｜></think>", 0.0f, 1.05f},
        {"Repetition Penalty 1.10", "<｜User｜>Repeat after me: /home/dpavlin/koha-rfid-go/internal/rfid/rfid501.go<｜Assistant｜></think>", 0.0f, 1.10f},
        {"Repetition Penalty 1.20", "<｜User｜>Repeat after me: /home/dpavlin/koha-rfid-go/internal/rfid/rfid501.go<｜Assistant｜></think>", 0.0f, 1.20f}
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
        req.n_gen = 45;
        req.sampler.temp = tc.temp;
        req.sampler.rep_pen = tc.rep_penalty;

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
