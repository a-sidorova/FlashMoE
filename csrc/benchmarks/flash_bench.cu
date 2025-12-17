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
    printf("epRank: %u : %f %f %f %f %f %f %f %f\n", flashmoe::hostBookkeeping.rank, *(fHp), *(fHp + 1), *(fHp + 2), *(fHp + 3), *(fHp + 4), *(fHp + 5), *(fHp + 6), *(fHp + 7));

    float timed = 0;
    printf("forwardHost\n");
    flashmoe::moe::forwardHost(p, p + dZ * sizeof(Element));
    //printf("epRank: %u took %.2fms\n", flashmoe::hostBookkeeping.rank, timed);
    auto* hO = std::calloc(S * PX + S * H, sizeof(float));
    auto* fHo = static_cast<float*>(hO);
    auto* __restrict__ eHo = static_cast<Element*>(hO);
    FLASHMOE_CHECK_CUDA(cudaMemcpy(eHo, p + dZ * sizeof(Element), sizeof(Element) * (S * PX + S * H),
        cudaMemcpyDeviceToHost));
    // Wait for the async D2H copy to complete before reading host buffer.
    //FLASHMOE_CHECK_CUDA(cudaStreamSynchronize(flashmoe::flashmoeStream));
    //fHo = fHo + S * PX;
    printf("epRank: %u : %f %f %f %f %f %f %f %f\n", flashmoe::hostBookkeeping.rank, *(fHo), *(fHo + 1), *(fHo + 2), *(fHo + 3), *(fHo + 4), *(fHo + 5), *(fHo + 6), *(fHo + 7));

    std::cout << std::endl;
    FLASHMOE_CHECK_CUDA(cudaPeekAtLastError());
    flashmoe::finalize();
    std::free(hP);
    std::free(hO);
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

    auto* hO = std::calloc(S * PX + S * H, sizeof(float));
    auto* fHo = static_cast<float*>(hO);
    auto* __restrict__ eHo = static_cast<Element*>(hO);

    auto gateOutputSize = gZ - dZ;
    auto moeOutputSize = cZ - gZ;
    auto gateOutputMemPtr = std::calloc(gateOutputSize, sizeof(float));
    auto moeOutputMemPtr = std::calloc(moeOutputSize, sizeof(float));

    auto* fGateOutputMemPtr = static_cast<float*>(gateOutputMemPtr);
    auto* fMoeOutputMemPtr = static_cast<float*>(moeOutputMemPtr);

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

        // expert copy own weights
        std::memcpy(fHp + gwZ, expertWeights.data() + rank * 2 * nLx * (P * H), 2 * nLx * (P * H) * sizeof(float));
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
        std::vector<float> gateOutput(S * PX, 0);
        std::vector<float> moeOutput(S * H, 0);

        std::memcpy(activations.data(), fHp, (S * H) * sizeof(float));
        std::memcpy(gateWeights.data(), fHp + S * H, (PX * H) * sizeof(float));

        printf("Forward for Rank: %u \n", flashmoe::hostBookkeeping.rank);
        flashmoe::forwardCPU<S, H, P, PX, E>(activations, gateWeights, expertWeights, gateOutput, moeOutput);

        bool failed = false;
        for (size_t i = 0; i < gateOutputSize; ++i) {
            if (std::abs(fGateOutputMemPtr[i] - gateOutput[i]) > 1e3) {
                printf("Elementwise difference of softmax (gate + softmax) outputs for Rank: %u: Error at index %zu: %f vs %f\n",
                        flashmoe::hostBookkeeping.rank, i, fGateOutputMemPtr[i], gateOutput[i]);
                failed = true;
                break;
            }
        }
        if (!failed) {
            printf("Elementwise difference has not been found for Rank: %u!\n", flashmoe::hostBookkeeping.rank);
        }
    }

    FLASHMOE_CHECK_CUDA(cudaPeekAtLastError());
    flashmoe::finalize();
    std::free(hP);
    std::free(hO);
}
