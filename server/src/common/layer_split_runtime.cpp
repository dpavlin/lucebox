#include "layer_split_runtime.h"

namespace dflash::common {

bool run_layer_split_ar_decode(
        int last_tok,
        int committed,
        int n_gen,
        int vocab,
        const std::vector<float> & prefill_last_logits,
        const SamplerCfg & sampler,
        std::mt19937_64 & rng,
        const std::vector<int32_t> & history_prefix,
        const LayerSplitForwardStep & forward_one,
        const std::function<bool(int)> & is_eos,
        std::vector<int32_t> & out_tokens,
        const DaemonIO & io) {
    if (n_gen <= 0) return true;

    std::vector<int32_t> history;
    if (sampler.needs_logit_processing()) {
        history.reserve(history_prefix.size() + out_tokens.size() + (size_t)n_gen);
        history.insert(history.end(), history_prefix.begin(), history_prefix.end());
        history.insert(history.end(), out_tokens.begin(), out_tokens.end());
        if ((int)prefill_last_logits.size() != vocab) return false;
        last_tok = sample_logits(prefill_last_logits.data(), vocab, sampler,
                                 history, rng);
    } else if (!prefill_last_logits.empty()) {
        float max_val = prefill_last_logits[0];
        int max_idx = 0;
        for (int i = 1; i < (int)prefill_last_logits.size(); i++) {
            if (prefill_last_logits[i] > max_val) {
                max_val = prefill_last_logits[i];
                max_idx = i;
            }
        }
        last_tok = max_idx;
    }

    out_tokens.push_back(last_tok);
    if (sampler.needs_logit_processing()) history.push_back(last_tok);
    io.emit(last_tok);
    if (io.is_cancelled()) {
        io.emit(-1);
        return true;
    }
    if (is_eos(last_tok)) {
        io.emit(-1);
        return true;
    }
    ++committed;

    std::vector<float> logits_buf;
    for (int i = 1; i < n_gen; ++i) {
        std::vector<int32_t> one(1, last_tok);
        int next_tok = -1;
        logits_buf.clear();
        if (!forward_one(one, committed, next_tok, &logits_buf)) {
            return false;
        }
        if (sampler.needs_logit_processing()) {
            if ((int)logits_buf.size() != vocab) return false;
            next_tok = sample_logits(logits_buf.data(), vocab, sampler,
                                     history, rng);
        } else if (!logits_buf.empty()) {
            float max_val = logits_buf[0];
            int max_idx = 0;
            for (int j = 1; j < (int)logits_buf.size(); j++) {
                if (logits_buf[j] > max_val) {
                    max_val = logits_buf[j];
                    max_idx = j;
                }
            }
            next_tok = max_idx;
        }

        if (std::getenv("DFLASH_DEBUG_LOGITS") && !logits_buf.empty()) {
            float logit_0 = (logits_buf.size() > 18) ? logits_buf[18] : 0.0f;
            float logit_o = (logits_buf.size() > 81) ? logits_buf[81] : 0.0f;
            std::vector<std::pair<float, int32_t>> topk;
            for (int k = 0; k < (int)logits_buf.size(); k++) topk.push_back({logits_buf[k], k});
            std::partial_sort(topk.begin(), topk.begin() + std::min((size_t)5, topk.size()), topk.end(),
                              [](const auto & a, const auto & b) { return a.first > b.first; });
            std::fprintf(stderr, "[debug-logits] i=%d pos=%d next=%d ('0'#18=%.3f, 'o'#81=%.3f, diff=%.3f) top5=[",
                         i, committed, next_tok, logit_0, logit_o, logit_o - logit_0);
            for (size_t k = 0; k < 5 && k < topk.size(); k++) {
                std::fprintf(stderr, "(id=%d, l=%.3f)%s", topk[k].second, topk[k].first, k + 1 < 5 ? ", " : "");
            }
            std::fprintf(stderr, "]\n");
            std::fflush(stderr);
        }

        last_tok = next_tok;
        out_tokens.push_back(last_tok);
        if (sampler.needs_logit_processing()) history.push_back(last_tok);
        io.emit(last_tok);
        ++committed;
        if (io.is_cancelled()) break;
        if (is_eos(last_tok)) break;
    }

    io.emit(-1);
    return true;
}

}  // namespace dflash::common
