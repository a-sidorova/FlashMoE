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
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>
#include <omp.h>
#include "../include/flashmoe/types.cuh"

namespace flashmoe {
    template<
        unsigned int K,
        unsigned int N
    >
    void matmul(const float* a,
                const float* b,
                float* c,
                size_t M,
                bool transposed_b) {
#pragma omp parallel for
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
        unsigned int PX,
        unsigned int E
    >
    void softmax(std::vector<float>& gateOutput, size_t S) {
        std::vector<float> gateSoftmax(gateOutput.size(), 0.0f);
#pragma omp parallel for
        for (size_t s = 0; s < S; ++s) {
            // Find max for numerical stability
            float max_val = gateOutput[s * PX];
            for (size_t e = 1; e < E; ++e) {
                max_val = std::max(max_val, gateOutput[s * PX + e]);
            }
            // Compute exp and sum
            float exp_sum = 0.0f;
            for (size_t e = 0; e < E; ++e) {
                gateSoftmax[s * PX + e] = std::exp(gateOutput[s * PX + e] - max_val);
                exp_sum += gateSoftmax[s * PX + e];
            }
            // Normalize
            for (size_t e = 0; e < E; ++e) {
                gateSoftmax[s * PX + e] /= exp_sum;
            }
        }
        gateOutput = gateSoftmax;
    }

    template<
        unsigned int PX,
        unsigned int E,
        unsigned int K
    >
    void topK(const std::vector<float>& softmax, std::vector<float>& topk, size_t S) {
        for (size_t s = 0; s < S; ++s) {
            std::vector<std::pair<float, size_t>> e_scores;
            for (size_t e = 0; e < E; ++e) {
                e_scores.emplace_back(softmax[s * PX + e], e);
            }
            std::sort(
                e_scores.begin(),
                e_scores.end(),
                [](const auto& a, const auto& b) { return a.first > b.first; }
            );
            for (size_t k = 0; k < K; ++k) {
                topk[s * K + k] = e_scores[k].second;
            }
        }
    }

    inline float activation(float x) {
        if constexpr (flashmoe::ACC::HA::value == 0U) { // ReLU
            return std::max(0.0f, x);
        } else if constexpr (flashmoe::ACC::HA::value == 1U) { // GELU (approx)
            constexpr float pi2 = 2.0f / M_PI;
            float kAlpha = std::sqrt(pi2);
            constexpr float kBeta = 0.044715f;
            return 0.5f * x * (1.0f + std::tanh(kAlpha * (x + kBeta * x * x * x)));
        } else {
            return x;
        }
    }

    template<
        unsigned int H,
        unsigned int P,
        unsigned int PX,
        unsigned int E
    >
    void experts(const std::vector<float>& activations,
                 const std::vector<float>& expertWeights,
                 const std::vector<float>& topk,
                 const std::vector<float>& probs,
                 std::vector<float>& moeOutput,
                 size_t S) {
        // Bucket tokens per expert based on routing results
        std::vector<std::vector<size_t>> expertTokens(E);
        const auto K = topk.size() / S;
        for (size_t s = 0; s < S; ++s) {
            for (size_t k = 0; k < K; ++k) {
                const auto expertIdx = static_cast<size_t>(topk[s * K + k]);
                if (expertIdx >= E) {
                    continue; // ignore padded experts
                }
                expertTokens[expertIdx].push_back(s);
            }
        }

        // Compute expert outputs using matmul and scatter back
        for (size_t e = 0; e < E; ++e) {
            const auto tokenCount = expertTokens[e].size();
            if (!tokenCount) {
                continue;
            }

            // Slice expert weights: [P x H] then [H x P]
            const float* wUp = expertWeights.data() + e * 2 * P * H;
            const float* wDown = wUp + P * H;

            // Gather activations for this expert
            std::vector<float> aExpert(tokenCount * H);
            for (size_t i = 0; i < tokenCount; ++i) {
                const auto tokenIdx = expertTokens[e][i];
                std::copy_n(activations.data() + tokenIdx * H, H, aExpert.data() + i * H);
            }

            // Up projection: [tokenCount x H] * [P x H]^T -> [tokenCount x P]
            std::vector<float> hidden(tokenCount * P, 0.0f);
            matmul<H, P>(aExpert.data(), wUp, hidden.data(), tokenCount, true);

            // Activation
            for (auto& v : hidden) {
                v = activation(v);
            }

            // Down projection: [tokenCount x P] * [H x P]^T -> [tokenCount x H]
            std::vector<float> expertOut = aExpert;
            std::fill(expertOut.begin(), expertOut.end(), 0);
            matmul<P, H>(hidden.data(), wDown, expertOut.data(), tokenCount, true);

            // Scatter back with gate probability weighting
            for (size_t i = 0; i < tokenCount; ++i) {
                const auto tokenIdx = expertTokens[e][i];
                if (K > 1) {
                    const float gateProb = probs[tokenIdx * PX + e];
                    if (gateProb == 0.0f) {
                        continue;
                    }
                    for (size_t h = 0; h < H; ++h) {
                        moeOutput[tokenIdx * H + h] += gateProb * expertOut[i * H + h];
                    }
                } else {
                    for (size_t h = 0; h < H; ++h) {
                        moeOutput[tokenIdx * H + h] = expertOut[i * H + h];
                    }
                }
            }
        }
    }

    template<
        unsigned int H,
        unsigned int P,
        unsigned int PX,
        unsigned int E
    >
    void forwardCPU(const std::vector<float>& activations,
                    const std::vector<float>& gateWeights,
                    const std::vector<float>& expertWeights,
                    std::vector<float>& gateOutput,
                    std::vector<float>& moeOutput,
                    size_t S) {
        // Gate
        matmul<H, PX>(activations.data(), gateWeights.data(), gateOutput.data(), S, true);

        // Softmax
        softmax<PX, E>(gateOutput, S);

        // topK
        constexpr size_t K = flashmoe::ACC::TK::value;
        std::vector<float> topk(S * K, 0);
        topK<PX, E, K>(gateOutput, topk, S);

        // Expert computation
        experts<H, P, PX, E>(activations, expertWeights, topk, gateOutput, moeOutput, S);
    }
}
#endif //CORRECTNESS_CUH
