/******************************************************************************
 * Copyright (c) 2024, Osayamen Jonathan Aimuyo.
 ******************************************************************************/
#include <fmt/ranges.h>
#include <thrust/generate.h>
#include <thrust/random.h>

#include "../include/flashmoe/flashmoe.cuh"
#include "../correctness/correctness.cuh"

__host__ __forceinline__
void runOS() {
    flashmoe::initialize();
    const auto rank = flashmoe::getRank();
    // generate random input tile and eye weights
    constexpr auto S = flashmoe::ACC::S::value;
    constexpr auto H = flashmoe::ACC::H::value;
    constexpr auto E = flashmoe::ACC::E::value;
    constexpr auto P = flashmoe::ACC::P::value;
    constexpr auto PX = flashmoe::ACC::PX::value;
    const auto nLx = flashmoe::hostBookkeeping.nLx;
    constexpr unsigned long aZ =  S * H;
    constexpr auto gwZ = aZ + PX * H;
    // scale this to number of experts
    const auto bZ =  gwZ + nLx * P * H;
    const auto b2Z =  bZ + nLx * P * H;
    const auto dZ =  b2Z + nLx * (P + H);
    const auto gZ = dZ + S * PX;
    const auto cZ = gZ + S * H;
    cuda::std::byte* p;
    FLASHMOE_CHECK_CUDA(cudaMallocAsync(&p, cZ * sizeof(float), flashmoe::flashmoeStream));
    FLASHMOE_CHECK_CUDA(cudaMemsetAsync(p, 0, cZ * sizeof(float), flashmoe::flashmoeStream));
    auto* hP = std::calloc(cZ, sizeof(float));
    auto* fHp = static_cast<float*>(hP);
    using Element = flashmoe::ACC::Element;
    auto* __restrict__ eHp = static_cast<Element*>(hP);
    {
        #if FLASHMOE_NVTX
        flashmoe::flashmoeRange forwardRange{"Host Data Prep"};
        #endif
        thrust::default_random_engine rng(47 * (rank + 42));
        thrust::normal_distribution<float> dist(0, 5);
        // Activations
        thrust::generate(fHp, fHp + aZ, [&] { return dist(rng); });
        // gate weights
        thrust::generate(fHp + aZ, fHp + aZ + E * H, [&] { return dist(rng); });
        // Expert weights
        // loop for number of experts
        for (uint i = 0; i < nLx; ++i) {
            // expert up
            thrust::generate(fHp + gwZ + i * (P * H), fHp + gwZ + (i + 1) * (P * H),
                [&] { return dist(rng); });
            thrust::generate(fHp + bZ + i * (P * H), fHp + bZ + (i + 1) * (P * H),
                [&] { return dist(rng); });
        }
        // bias
        std::ranges::fill(fHp + b2Z, fHp + dZ, 0.0f);
        constexpr cutlass::NumericConverter<Element, float> conv{};
        for (uint i = 0; i < dZ; ++i) {
            eHp[i] = conv(fHp[i]);
        }
    }
    FLASHMOE_CHECK_CUDA(cudaMemcpyAsync(p, eHp, sizeof(Element) * dZ,
        cudaMemcpyHostToDevice,
        flashmoe::flashmoeStream));
    float timed = 0;
    printf("forwardHost\n");
    flashmoe::moe::forwardHost(p, p + dZ * sizeof(Element));

    auto gateOutputSize = gZ - dZ;
    auto moeOutputSize = cZ - gZ;
    auto gateOutputMemPtr = std::calloc(gateOutputSize, sizeof(float));
    auto moeOutputMemPtr = std::calloc(moeOutputSize, sizeof(float));

    auto* fGateOutputMemPtr = static_cast<float*>(gateOutputMemPtr);
    auto* fMoeOutputMemPtr = static_cast<float*>(moeOutputMemPtr);

    FLASHMOE_CHECK_CUDA(cudaStreamSynchronize(flashmoe::flashmoeStream));
    FLASHMOE_CHECK_CUDA(cudaMemcpy(fGateOutputMemPtr, p + dZ * sizeof(Element), gateOutputSize * sizeof(float),
        cudaMemcpyDeviceToHost));
    FLASHMOE_CHECK_CUDA(cudaMemcpy(fMoeOutputMemPtr, p + gZ * sizeof(Element), moeOutputSize * sizeof(float),
        cudaMemcpyDeviceToHost));

    FLASHMOE_CHECK_CUDA(cudaPeekAtLastError());
    flashmoe::finalize();
    std::free(hP);
    std::free(fGateOutputMemPtr);
    std::free(fMoeOutputMemPtr);
}

int main() {
    using Element = flashmoe::ACC::Element;

    flashmoe::initialize();
    const auto rank = flashmoe::getRank();
    // generate random input tile and eye weights
    constexpr auto S = flashmoe::ACC::S::value;
    constexpr auto H = flashmoe::ACC::H::value;
    constexpr auto E = flashmoe::ACC::E::value;
    constexpr auto P = flashmoe::ACC::P::value;
    constexpr auto PX = flashmoe::ACC::PX::value;
    const auto nLx = flashmoe::hostBookkeeping.nLx;
    constexpr unsigned long aZ =  S * H;
    constexpr auto gwZ = aZ + PX * H;
    // scale this to number of experts
    const auto bZ =  gwZ + nLx * P * H;
    const auto b2Z =  bZ + nLx * P * H;
    const auto dZ =  b2Z + nLx * (P + H);
    const auto gZ = dZ + S * PX;
    const auto cZ = gZ + S * H;
    cuda::std::byte* p;
    FLASHMOE_CHECK_CUDA(cudaMallocAsync(&p, cZ * sizeof(float), flashmoe::flashmoeStream));
    FLASHMOE_CHECK_CUDA(cudaMemsetAsync(p, 0, cZ * sizeof(float), flashmoe::flashmoeStream));

    auto* hP = std::calloc(cZ, sizeof(float));
    auto* fHp = static_cast<float*>(hP);
    auto* __restrict__ eHp = static_cast<Element*>(hP);

    auto gateOutputSize = gZ - dZ;
    auto moeOutputSize = cZ - gZ;
    auto gateOutputMemPtr = std::calloc(gateOutputSize, sizeof(float));
    auto moeOutputMemPtr = std::calloc(moeOutputSize, sizeof(float));

    auto* fGateOutputMemPtr = static_cast<float*>(gateOutputMemPtr);
    auto* fMoeOutputMemPtr = static_cast<float*>(moeOutputMemPtr);

    auto refGateOutput = std::vector<float>(S * PX, 0);
    auto refMoeOutput = std::vector<float>(S * H, 0);

    assert(nLx == flashmoe::hostBookkeeping.world);

    // Generate common expert weights [E * 2 * P * H]
    std::vector<float> expertWeights(E * 2 * P * H);
    thrust::default_random_engine rng(131);
    thrust::normal_distribution<float> dist(0, 0.15);
    thrust::generate(expertWeights.data(), expertWeights.data() + expertWeights.size(),
                [&] { return dist(rng); });

    {
        #if FLASHMOE_NVTX
        flashmoe::flashmoeRange forwardRange{"Host Data Prep"};
        #endif
        thrust::default_random_engine rng(47 * (rank + 42));
        thrust::normal_distribution<float> dist(0, 0.15);
        // Activations
        thrust::generate(fHp, fHp + aZ, [&] { return dist(rng); });
        // gate weights
        thrust::generate(fHp + aZ, fHp + aZ + E * H, [&] { return dist(rng); });
        // bias
        std::ranges::fill(fHp + b2Z, fHp + dZ, 0.0f);

        // copy own expert weights
        for (uint i = 0; i < nLx; ++i) {
            std::memcpy(fHp + gwZ + i * (P * H), expertWeights.data() + (i + nLx * rank) * 2 * P * H, (P * H) * sizeof(float));
            std::memcpy(fHp + bZ + i * (P * H), expertWeights.data() + (i + nLx * rank) * 2 * P * H + P * H, (P * H) * sizeof(float));
        }
        constexpr cutlass::NumericConverter<Element, float> conv{};
        for (uint i = 0; i < dZ; ++i) {
            eHp[i] = conv(fHp[i]);
        }
    }

    printf("===== FlashMoE target execution =====\n");
    {
        FLASHMOE_CHECK_CUDA(cudaMemcpyAsync(p, eHp, sizeof(Element) * dZ,
        cudaMemcpyHostToDevice,
        flashmoe::flashmoeStream));

        printf("Forward for Rank: %u \n", flashmoe::hostBookkeeping.rank);
        flashmoe::moe::forwardHost(p, p + dZ * sizeof(Element));

        FLASHMOE_CHECK_CUDA(cudaStreamSynchronize(flashmoe::flashmoeStream));
        FLASHMOE_CHECK_CUDA(cudaMemcpy(fGateOutputMemPtr, p + dZ * sizeof(Element), gateOutputSize * sizeof(float),
            cudaMemcpyDeviceToHost));
        FLASHMOE_CHECK_CUDA(cudaMemcpy(fMoeOutputMemPtr, p + gZ * sizeof(Element), moeOutputSize * sizeof(float),
            cudaMemcpyDeviceToHost));
    }

    printf("===== FlashMoE reference execution =====\n");
    {
        auto rankCount = flashmoe::hostBookkeeping.world;
        std::vector<float> activations(S * H);
        std::vector<float> gateWeights(PX * H);

        std::memcpy(activations.data(), fHp, (S * H) * sizeof(float));
        std::memcpy(gateWeights.data(), fHp + S * H, (PX * H) * sizeof(float));

        printf("Forward for Rank: %u \n", flashmoe::hostBookkeeping.rank);
        flashmoe::forwardCPU<H, P, PX, E>(activations, gateWeights, expertWeights, refGateOutput, refMoeOutput, S);
    }

    printf("===== Validation =====\n");
    {
        bool failed = false;
        for (size_t i = 0; i < gateOutputSize; ++i) {
            if (std::abs(fGateOutputMemPtr[i] - refGateOutput[i]) > 1e-2) {
                printf("Elementwise difference of softmax (gate + softmax) outputs for Rank: %u: Error at index %zu: %f vs %f\n",
                        flashmoe::hostBookkeeping.rank, i, fGateOutputMemPtr[i], refGateOutput[i]);
                failed = true;
                break;
            }
        }
        if (!failed) {
            printf("Elementwise difference between softmax (gate + softmax) outputs and reference has not been found for Rank: %u!\n", flashmoe::hostBookkeeping.rank);
        }

        failed = false;
        for (size_t i = 0; i < moeOutputSize; ++i) {
            if (std::abs(fMoeOutputMemPtr[i] - refMoeOutput[i]) > 1e-3) {
                printf("Elementwise difference of entire MoE outputs for Rank: %u: Error at index %zu: %f vs %f\n",
                        flashmoe::hostBookkeeping.rank, i, fMoeOutputMemPtr[i], refMoeOutput[i]);
                failed = true;
                break;
            }
        }
        if (!failed) {
            printf("Elementwise difference between MoE outputs and reference has not been found for Rank: %u!\n", flashmoe::hostBookkeeping.rank);
        }
    }

    FLASHMOE_CHECK_CUDA(cudaPeekAtLastError());
    flashmoe::finalize();
    std::free(hP);
    std::free(fGateOutputMemPtr);
    std::free(fMoeOutputMemPtr);
}
