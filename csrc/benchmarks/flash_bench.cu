/******************************************************************************
 * Copyright (c) 2024, Osayamen Jonathan Aimuyo.
 ******************************************************************************/
#include <fmt/ranges.h>
#include <thrust/generate.h>
#include <thrust/random.h>

#include <fstream>

#include "../include/flashmoe/flashmoe.cuh"
#include "../correctness/correctness.cuh"

__host__ void saveMatrixToFile(const float* data,
    size_t rows,
    size_t cols,
    const std::string& bufferName,
    unsigned int rank) {
    const std::string filename = bufferName + "_rank" + std::to_string(rank) + ".txt";
    std::ofstream out(filename, std::ios::out | std::ios::trunc);
    if (!out.good()) {
        return;
    }
    out.setf(std::ios::fixed);
    out << std::setprecision(3);
    out << std::right;

    {
        std::ostringstream moeInfo;
        moeInfo << "rank=" << rank << "\n"
                  << "S=" << flashmoe::ACC::S::value << "\n"
                  << "H=" << flashmoe::ACC::H::value << "\n"
                  << "E=" << flashmoe::ACC::E::value << "\n"
                  << "P=" << flashmoe::ACC::P::value << "\n"
                  << "PX=" << flashmoe::ACC::PX::value << " (ceil_div(E, BLOCK_N) * BLOCK_N))\n"
                  << "nLx=" << flashmoe::hostBookkeeping.nLx << "\n"
                  << "matrix=" << bufferName << "\n"
                  << "shape=" << rows << "x" << cols << "\n";
        out << moeInfo.str() << "\n";
    }

    for (size_t c = 0; c < cols; ++c) {
        out << std::setw(6) << c;
        if (c + 1 < cols) {
            out << ' ';
        }
    }
    out << "\n";

    for (size_t r = 0; r < rows; ++r) {
        const size_t rowOffset = r * cols;
        for (size_t c = 0; c < cols; ++c) {
            out << std::setw(6) << data[rowOffset + c];
            if (c + 1 < cols) {
                out << ' ';
            }
        }
        out << "\n";
    }
    std::cout << "Saved to " << filename << "\n";
}

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
    constexpr auto TK = flashmoe::ACC::TK::value;
    const auto nLx = flashmoe::hostBookkeeping.nLx;
    constexpr unsigned long aZ =  S * H;
    constexpr auto gwZ = aZ + PX * H;
    // scale this to number of experts
    const auto bZ =  gwZ + nLx * P * H;
    const auto b2Z =  bZ + nLx * P * H;
    const auto dZ =  b2Z + nLx * (P + H);
    const auto gZ = dZ + S * PX;
    const auto oZ = gZ + S * H;
    const auto cZ = oZ + S * (2 * TK);
    cuda::std::byte* p;
    FLASHMOE_CHECK_CUDA(cudaMallocAsync(&p, cZ * sizeof(float), flashmoe::flashmoeStream));
    FLASHMOE_CHECK_CUDA(cudaMemsetAsync(p, 0, cZ * sizeof(float), flashmoe::flashmoeStream));

    auto* hP = std::calloc(cZ, sizeof(float));
    auto* fHp = static_cast<float*>(hP);
    auto* __restrict__ eHp = static_cast<Element*>(hP);

    auto gateOutputSize = gZ - dZ;
    auto topkOutputSize = cZ - oZ;
    auto moeOutputSize = oZ - gZ;
    auto gateOutputMemPtr = std::calloc(gateOutputSize, sizeof(float));
    auto TopKOutputMemPtr = std::calloc(topkOutputSize, sizeof(float));
    auto moeOutputMemPtr = std::calloc(moeOutputSize, sizeof(float));

    auto* fGateOutputMemPtr = static_cast<float*>(gateOutputMemPtr);
    auto* fTopkOutputMemPtr = static_cast<float*>(TopKOutputMemPtr);
    auto* fMoeOutputMemPtr = static_cast<float*>(moeOutputMemPtr);

    auto refGateOutput = std::vector<float>(S * PX, 0);
    auto refTopkOutput = std::vector<float>(S * (2 * TK), 0);
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

        printf("Forward for Rank: %u with local experts: %u\n", flashmoe::hostBookkeeping.rank, nLx);
        flashmoe::moe::forwardHost(p, p + dZ * sizeof(Element));

        FLASHMOE_CHECK_CUDA(cudaStreamSynchronize(flashmoe::flashmoeStream));
        FLASHMOE_CHECK_CUDA(cudaMemcpy(fGateOutputMemPtr, p + dZ * sizeof(Element), gateOutputSize * sizeof(float),
            cudaMemcpyDeviceToHost));
        FLASHMOE_CHECK_CUDA(cudaMemcpy(fMoeOutputMemPtr, p + gZ * sizeof(Element), moeOutputSize * sizeof(float),
            cudaMemcpyDeviceToHost));
        FLASHMOE_CHECK_CUDA(cudaMemcpy(fTopkOutputMemPtr, p + oZ * sizeof(Element), topkOutputSize * sizeof(float),
            cudaMemcpyDeviceToHost));

        printf("Output Gate for epRank %u : %f %f %f %f %f \n", flashmoe::hostBookkeeping.rank,
            fGateOutputMemPtr[0], fGateOutputMemPtr[1], fGateOutputMemPtr[2], fGateOutputMemPtr[3], fGateOutputMemPtr[4]);
        printf("Output MoE for epRank %u : %f %f %f %f %f \n", flashmoe::hostBookkeeping.rank,
                fMoeOutputMemPtr[0], fMoeOutputMemPtr[1], fMoeOutputMemPtr[2], fMoeOutputMemPtr[3], fMoeOutputMemPtr[4]);
    }

    printf("===== FlashMoE reference execution =====\n");
    {
        auto rankCount = flashmoe::hostBookkeeping.world;
        std::vector<float> activations(S * H);
        std::vector<float> gateWeights(PX * H);

        std::memcpy(activations.data(), fHp, (S * H) * sizeof(float));
        std::memcpy(gateWeights.data(), fHp + S * H, (PX * H) * sizeof(float));

        printf("Forward for Rank: %u \n", flashmoe::hostBookkeeping.rank);
        flashmoe::forwardCPU<H, P, PX, E>(activations, gateWeights, expertWeights, refGateOutput, refTopkOutput, refMoeOutput, S);

        printf("Output Gate for epRank %u : %f %f %f %f %f \n", flashmoe::hostBookkeeping.rank,
            refGateOutput[0], refGateOutput[1], refGateOutput[2], refGateOutput[3], refGateOutput[4]);
        printf("Output MoE for epRank %u : %f %f %f %f %f \n", flashmoe::hostBookkeeping.rank,
            refMoeOutput[0], refMoeOutput[1], refMoeOutput[2], refMoeOutput[3], refMoeOutput[4]);
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
        for (size_t i = 0; i < topkOutputSize; ++i) {
            if (std::abs(fTopkOutputMemPtr[i] - refTopkOutput[i]) > 1e-3) {
                printf("Elementwise difference of TopK outputs for Rank: %u: Error at index %zu: %f vs %f\n",
                        flashmoe::hostBookkeeping.rank, i, fTopkOutputMemPtr[i], refTopkOutput[i]);
                failed = true;
                break;
            }
        }
        if (!failed) {
            printf("Elementwise difference between TopK outputs and reference has not been found for Rank: %u!\n", flashmoe::hostBookkeeping.rank);
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
