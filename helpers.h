/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <cstdio>
#include <stdexcept>
#include <vector>
#include <functional>

#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_fp8.h>
// #include <cuda_bf16.h>
// #include <cuda_fp16.h>

using namespace std;

#define INITTIMER            \
    float milliseconds = 0;  \
    cudaEvent_t start, stop; \
    cudaEventCreate(&start); \
    cudaEventCreate(&stop);
#define START cudaEventRecord(start);
#define END_wo_print            \
    cudaEventRecord(stop);      \
    cudaEventSynchronize(stop); \
    cudaEventElapsedTime(&milliseconds, start, stop);
#define END(x)                                        \
    cudaEventRecord(stop);                            \
    cudaEventSynchronize(stop);                       \
    cudaEventElapsedTime(&milliseconds, start, stop); \
    if (END_PRINT)                                    \
        printf("[+] %s Elapsed Time: %f ms\n", x, milliseconds);

inline void
checkCudaStatus(cudaError_t status)
{
    if (status != cudaSuccess)
    {
        printf("cuda API failed with status %d: %s\n", status, cudaGetErrorString(status));
        throw std::logic_error("cuda API failed");
    }
}

inline void checkCublasStatus(cublasStatus_t status)
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        printf("cuBLAS API failed with status %d\n", status);
        throw std::logic_error("cuBLAS API failed");
    }
}

template <typename InType, typename OutType = InType, typename ComputeType = OutType, typename BiasType = OutType>
struct TestBench
{
    using SampleRunner = std::function<void()>;

    TestBench(int m, int n, int k, ComputeType alpha = 0.0f, ComputeType beta = 0.0f, size_t workspaceSize = 1024 * 1024 * 4, int N = 1) : m(m), n(n), k(k), N(N), alpha(alpha), beta(beta), workspaceSize(workspaceSize), Ahost(m * k * N), Bhost(n * k * N),
                                                                                                                                           Chost(m * n * N), biasHost(m * N)
    {
        checkCublasStatus(cublasLtCreate(&ltHandle));
        checkCudaStatus(cudaMalloc(reinterpret_cast<void **>(&Adev), m * k * N * sizeof(InType)));
        checkCudaStatus(cudaMalloc(reinterpret_cast<void **>(&Bdev), n * k * N * sizeof(InType)));
        checkCudaStatus(cudaMalloc(reinterpret_cast<void **>(&Cdev), m * n * N * sizeof(OutType)));
        checkCudaStatus(cudaMalloc(reinterpret_cast<void **>(&biasDev), m * N * sizeof(BiasType)));
        checkCudaStatus(cudaMalloc(&workspace, workspaceSize));
        checkCudaStatus(cudaStreamCreate(&stream));

        fillData();
    }

    ~TestBench()
    {
        checkCublasStatus(cublasLtDestroy(ltHandle));
        checkCudaStatus(cudaFree(Adev));
        checkCudaStatus(cudaFree(Bdev));
        checkCudaStatus(cudaFree(Cdev));
        checkCudaStatus(cudaFree(biasDev));
        checkCudaStatus(cudaFree(workspace));
        checkCudaStatus(cudaStreamDestroy(stream));
    }

    void fillData()
    {
        for (int i = 0; i < m * k * N; i++)
            Ahost[i] = InType(i);
        for (int i = 0; i < n * k * N; i++)
            Bhost[i] = InType(i);
        for (int i = 0; i < m * N; i++)
            biasHost[i] = InType(i + 1);
    }

    void copyDataToDevice()
    {
        checkCudaStatus(cudaMemcpyAsync(Adev, Ahost.data(), Ahost.size() * sizeof(Ahost[0]), cudaMemcpyHostToDevice, stream));
        checkCudaStatus(cudaMemcpyAsync(Bdev, Bhost.data(), Bhost.size() * sizeof(Bhost[0]), cudaMemcpyHostToDevice, stream));
        checkCudaStatus(cudaMemcpyAsync(biasDev, biasHost.data(), biasHost.size() * sizeof(biasHost[0]), cudaMemcpyHostToDevice));
    }

    void copyDataFromDevice()
    {
        checkCudaStatus(cudaMemcpyAsync(Chost.data(), Cdev, Chost.size() * sizeof(Chost[0]), cudaMemcpyDeviceToHost, stream));
    }

    void streamSynchronize()
    {
        checkCudaStatus(cudaStreamSynchronize(stream));
    }

    void run(const SampleRunner &runSample)
    {
        copyDataToDevice();

        runSample();

        copyDataFromDevice();
        streamSynchronize();
    }

    int m, n, k, N;
    ComputeType alpha, beta;
    size_t workspaceSize;
    std::vector<InType> Ahost, Bhost;
    std::vector<OutType> Chost;
    std::vector<BiasType> biasHost;
    void *workspace;
    InType *Adev, *Bdev;
    OutType *Cdev;
    BiasType *biasDev;
    cudaStream_t stream;
    cublasLtHandle_t ltHandle;
};

template <>
inline void TestBench<__nv_fp8_e4m3, float, float, __nv_bfloat16>::fillData()
{
    for (int i = 0; i < m * k * N; i++)
        Ahost[i] = __nv_fp8_e4m3(i);
    for (int i = 0; i < n * k * N; i++)
        Bhost[i] = __nv_fp8_e4m3(i);
    for (int i = 0; i < m * N; i++)
        biasHost[i] = __float2bfloat16(i + 1);
}

template <>
inline void TestBench<__half>::fillData()
{
    for (int i = 0; i < m * k * N; i++)
        Ahost[i] = __float2half_rn(i);
    for (int i = 0; i < n * k * N; i++)
        Bhost[i] = __float2half_rn(i);
    for (int i = 0; i < m * N; i++)
        biasHost[i] = __float2half_rn(i + 1);
}
template <>
inline void TestBench<__half, __half, float>::fillData()
{
    for (int i = 0; i < m * k * N; i++)
        Ahost[i] = __float2half_rn(i);
    for (int i = 0; i < n * k * N; i++)
        Bhost[i] = __float2half_rn(i);
    for (int i = 0; i < m * N; i++)
        biasHost[i] = __float2half_rn(i + 1);
}

template <>
inline void TestBench<__half, __half, cuComplex>::fillData()
{
    for (int i = 0; i < m * k * N; i++)
        Ahost[i] = __float2half_rn(i / 100.);
    for (int i = 0; i < n * k * N; i++)
        Bhost[i] = __float2half_rn(i / 100.);
    for (int i = 0; i < m * N; i++)
        biasHost[i] = __float2half_rn(i + 1);
}