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

#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "cublasLt_Ltgemm_fp16.h"
#include "helpers.h"

/// Use cublasLtMatmul to perform tensor-op Cgemm using planar complex memory layout and half-precision inputs.
///
/// For better performance data order transforms should be offline as much as possible.
///
/// transa, transb assumed N; alpha, beta are host pointers, tensor ops allowed, alpha assumed 1, beta assumed 0,
/// stream assumed 0
/// outputs can be either single or half precision, half precision is used in this example
void Ltgemm_fp16(cublasLtHandle_t ltHandle,
                 int m,
                 int n,
                 int k,
                 const half *A,
                 int lda,
                 const half *B,
                 int ldb,
                 half *C,
                 int ldc)
{
    INITTIMER
    cublasLtMatmulDesc_t matmulDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
    half alpha = __float2half(1.0f);
    half beta = __float2half(0.0f);

    checkCublasStatus(cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_16F, CUDA_R_16F));

    // ---------------------------------------------------------------------------------------------
    // create descriptors for planar complex matrices

    checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_16F, m, k, lda));

    checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_16F, k, n, ldb));

    checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16F, m, n, ldc));

    // ---------------------------------------------------------------------------------------------
    // Launch computation
    START
    checkCublasStatus(cublasLtMatmul(ltHandle,
                                     matmulDesc,
                                     &alpha,
                                     A,
                                     Adesc,
                                     B,
                                     Bdesc,
                                     &beta,
                                     C,
                                     Cdesc,
                                     C,
                                     Cdesc,
                                     NULL,
                                     NULL,
                                     0,
                                     0));
    END_wo_print;
    printf("FP16-TensorCore M N K %d %d %d: %f ms\n", m, n, k, milliseconds);
    // descriptors are no longer needed as all GPU work was already enqueued
    if (Cdesc)
        checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc)
        checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc)
        checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
    if (matmulDesc)
        checkCublasStatus(cublasLtMatmulDescDestroy(matmulDesc));
}