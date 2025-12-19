/******************************************************************************
 * Copyright (c) 2024, Osayamen Jonathan Aimuyo.
 ******************************************************************************/
#include <fmt/ranges.h>
#include <thrust/generate.h>
#include <thrust/random.h>

#include "../include/flashmoe/flashmoe.cuh"

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
        thrust::normal_distribution<float> dist(0, 0.15);
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

    flashmoe::moe::forwardHost(p, p + dZ * sizeof(Element));
    printf("epRank: %u took %.2fms\n", flashmoe::hostBookkeeping.rank);

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

    printf("\nOutput Gate for epRank %u : %f %f %f %f %f \n", flashmoe::hostBookkeeping.rank, fGateOutputMemPtr[0], fGateOutputMemPtr[1], fGateOutputMemPtr[2], fGateOutputMemPtr[3], fGateOutputMemPtr[4]);
    printf("Output MoE for epRank %u : %f %f %f %f %f \n", flashmoe::hostBookkeeping.rank, fMoeOutputMemPtr[0], fMoeOutputMemPtr[1], fMoeOutputMemPtr[2], fMoeOutputMemPtr[3], fMoeOutputMemPtr[4]);

    FLASHMOE_CHECK_CUDA(cudaPeekAtLastError());
    flashmoe::finalize();
    std::free(hP);
}

int main() {
    runOS();
}
