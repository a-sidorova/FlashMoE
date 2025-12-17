/*
 * Copyright (c) 2025, Osayamen Jonathan Aimuyo
 * All rights reserved.
 *
 * This file is part of the Flashmoe Project and is licensed under the BSD 3-Clause License.
 * See the LICENSE file in the root directory for full terms.
 */

//
// Created by oja7 on 5/17/25.
//

#ifndef CORRECTNESS_CUH
#define CORRECTNESS_CUH
#include <numeric>
#include "../include/flashmoe/types.cuh"

namespace flashmoe {
    template<
        unsigned int M,
        unsigned int K,
        unsigned int N
    >
    void matmul(const std::vector<float>& a,
                const std::vector<float>& b,
                std::vector<float>&c,
                bool transposed_b) {
        for (size_t m = 0; m < M; ++m) {
            for (size_t n = 0; n < N; ++n) {
                float acc = 0.0f;
                for (size_t k = 0; k < K; ++k) {
                    acc += a[m * K + k] * b[transposed_b ? n * K + k : k * N + n];
                }
                c[m * N + n] = acc;
            }
        }
    }

    // Softmax along PX dimension for each token s
    template<
        unsigned int S,
        unsigned int PX
    >
    void softmax(std::vector<float>& gateOutput) {
        std::vector<float> gateSoftmax(gateOutput.size(), 0.0f);
        for (size_t s = 0; s < S; ++s) {
            // Find max for numerical stability
            float max_val = gateOutput[s * PX];
            for (size_t px = 1; px < PX; ++px) {
                max_val = std::max(max_val, gateOutput[s * PX + px]);
            }
            // Compute exp and sum
            float exp_sum = 0.0f;
            for (size_t px = 0; px < PX; ++px) {
                gateSoftmax[s * PX + px] = std::exp(gateOutput[s * PX + px] - max_val);
                exp_sum += gateSoftmax[s * PX + px];
            }
            // Normalize
            for (size_t px = 0; px < PX; ++px) {
                gateSoftmax[s * PX + px] /= exp_sum;
            }
        }
        gateOutput = gateSoftmax;
    }

    template<
        unsigned int S,
        unsigned int PX,
        unsigned int K
    >
    void topK(const std::vector<float>& softmax, std::vector<float>& topk) {
        for (size_t s = 0; s < S; ++s) {
            std::vector<std::pair<float, size_t>> px_scores;
            for (size_t px = 0; px < PX; ++px) {
                px_scores.emplace_back(softmax[s * PX + px], px);
            }
            std::partial_sort(
                px_scores.begin(),
                px_scores.begin() + K,
                px_scores.end(),
                [](const auto& a, const auto& b) { return a.first > b.first; }
            );
            for (size_t k = 0; k < K; ++k) {
                topk[s * K + k] = px_scores[k].second;
            }
        }
    }

    template<
        unsigned int S,
        unsigned int H,
        unsigned int P,
        unsigned int PX,
        unsigned int E
    >
    void forwardCPU(const std::vector<float>& activations,
                    const std::vector<float>& gateWeights,
                    const std::vector<float>& expertWeights,
                    std::vector<float>& gateOutput,
                    std::vector<float>& moeOutput) {
        // Gate
        matmul<S, H, PX>(activations, gateWeights, gateOutput, true);

        // Softmax
        softmax<S, PX>(gateOutput);

        // topK
        constexpr size_t K = flashmoe::ACC::TK::value;
        std::vector<float> topk(S * K, 0);
        topK<S, PX, K>(gateOutput, topk);
    }
}
#endif //CORRECTNESS_CUH
