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
#include <cstdint>

#include "cublasLt_Ltgemm_fp8.h"
#include "helpers.h"

int roundoff(int v, int d)
{
    return (v + d - 1) / d * d;
}

/// Use cublasLtMatmul to perform tensor-op Igemm with memory order transforms on all buffers
///
/// For better performance data order transforms should be offline as much as possible.
///
/// transa, transb assumed N; alpha, beta are host pointers, tensor ops allowed, alpha assumed 1, beta assumed 0,
/// stream assumed 0
void LtgemmTensor_fp8(cublasLtHandle_t ltHandle,
                   int m,
                   int n,
                   int k,
                   const float *alpha, /* host pointer */
                   const __nv_fp8_e4m3 *A,
                   int lda,
                   const __nv_fp8_e4m3 *B,
                   int ldb,
                   const float *beta, /* host pointer */
                   float *C,
                   int ldc,
                   void *workspace,
                   size_t workspaceSize)
{
    INITTIMER
    cublasLtMatmulDesc_t matmulDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
    // float alpha = 1.0f, beta = 0;
    cublasOperation_t opTranspose = CUBLAS_OP_T;

    // tensor op igemm kernels require specialized memory order of data
    cublasLtMatrixTransformDesc_t transformDesc = NULL;
    __nv_fp8_e4m3 *Atransform = NULL, *Btransform = NULL;
    float *Ctransform = NULL;
    cublasLtMatrixLayout_t AtransformDesc = NULL, BtransformDesc = NULL, CtransformDesc = NULL;
    float transformAlpha = 1.0f, transformBeta = 0.0f;
    cublasLtOrder_t order_COL32 = CUBLASLT_ORDER_COL32;
    cublasLtOrder_t order_COL4_4R2_8C = CUBLASLT_ORDER_COL4_4R2_8C;
    cublasLtOrder_t order_COL32_2R_4R4 = CUBLASLT_ORDER_COL32_2R_4R4;
    int ldatransform = 32 * m;
    int ldbtransform = 32 * roundoff(n, 8);
    int ldctransform = 32 * m;

    checkCudaStatus(cudaMalloc(reinterpret_cast<void **>(&Atransform), sizeof(__nv_fp8_e4m3) * roundoff(k, 32) / 32 * ldatransform));
    checkCudaStatus(cudaMalloc(reinterpret_cast<void **>(&Btransform), sizeof(__nv_fp8_e4m3) * roundoff(k, 32) / 32 * ldbtransform));
    checkCudaStatus(cudaMalloc(reinterpret_cast<void **>(&Ctransform), sizeof(float) * roundoff(n, 32) / 32 * ldctransform));

    checkCublasStatus(cublasLtMatrixTransformDescCreate(&transformDesc, CUDA_R_32F));

    checkCublasStatus(cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    // tensor op igemm kernels only support NT gemm
    checkCublasStatus(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opTranspose, sizeof(opTranspose)));

    // ---------------------------------------------------------------------------------------------
    // create descriptors for original matrices

    checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8F_E4M3, m, k, lda));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8F_E4M3, k, n, ldb));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, m, n, ldc));

    // ---------------------------------------------------------------------------------------------
    // create descriptors for transformed matrices

    checkCublasStatus(cublasLtMatrixLayoutCreate(&AtransformDesc, CUDA_R_8F_E4M3, m, k, ldatransform));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(AtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32, sizeof(order_COL32)));

    // data memory order is set to CUBLASLT_ORDER_COL4_4R2_8C in order to achieve best performance on Turing devices.
    // for best performance on Ampere, consider setting the memory order to CUBLASLT_ORDER_COL32_2R_4R4.
    checkCublasStatus(cublasLtMatrixLayoutCreate(&BtransformDesc, CUDA_R_8F_E4M3, n, k, ldbtransform));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(BtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32_2R_4R4, sizeof(order_COL32_2R_4R4)));

    checkCublasStatus(cublasLtMatrixLayoutCreate(&CtransformDesc, CUDA_R_32F, m, n, ldctransform));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(CtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32, sizeof(order_COL32)));

    // ---------------------------------------------------------------------------------------------
    // transforms and computation

    /*NOT SUPPORT*/
    /*LTZHANG-TODO: Uncomment when SUPPORT*/
    // checkCublasStatus(cublasLtMatrixTransform(ltHandle, transformDesc, &transformAlpha, A, Adesc, &transformBeta, NULL, NULL, Atransform, AtransformDesc, 0));

    // B matrix is non-transposed, but transposed matrix is needed - add transpose operation in matrix transform.
    checkCublasStatus(cublasLtMatrixTransformDescSetAttribute(transformDesc, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &opTranspose, sizeof(opTranspose)));

    /*NOT SUPPORT*/
    /*LTZHANG-TODO: Uncomment when SUPPORT*/
    //checkCublasStatus(cublasLtMatrixTransform(ltHandle, transformDesc, &transformAlpha, B, Bdesc, &transformBeta, NULL, NULL, Btransform, BtransformDesc, 0));

    START
    // no need to transform C matrix as beta is assumed to be 0
    checkCublasStatus(cublasLtMatmul(ltHandle,
                                     matmulDesc,
                                     &alpha,
                                     Atransform,
                                     AtransformDesc,
                                     Btransform,
                                     BtransformDesc,
                                     &beta,
                                     Ctransform,
                                     CtransformDesc,
                                     Ctransform,
                                     CtransformDesc,
                                     NULL,
                                     /*workspace,
                                     workspaceSize,
                                     ->
                                     NULL,
                                     NULL,*/
                                     workspace,
                                     workspaceSize,
                                     0));

    opTranspose = CUBLAS_OP_N;
    checkCublasStatus(cublasLtMatrixTransformDescSetAttribute(transformDesc, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &opTranspose, sizeof(opTranspose)));

    // transform outputs to COL order
    checkCublasStatus(cublasLtMatrixTransform(ltHandle, transformDesc, &transformAlpha, Ctransform, CtransformDesc, &transformBeta, NULL, NULL, C, Cdesc, 0));
    END_wo_print;
    printf("FP8-TensorCore M N K %d %d %d: %f ms\n", m, n, k, milliseconds);

    // descriptors are no longer needed as all GPU work was already enqueued
    if (CtransformDesc)
        checkCublasStatus(cublasLtMatrixLayoutDestroy(CtransformDesc));
    if (BtransformDesc)
        checkCublasStatus(cublasLtMatrixLayoutDestroy(BtransformDesc));
    if (AtransformDesc)
        checkCublasStatus(cublasLtMatrixLayoutDestroy(AtransformDesc));
    if (Cdesc)
        checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc)
        checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc)
        checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
    if (matmulDesc)
        checkCublasStatus(cublasLtMatmulDescDestroy(matmulDesc));
    if (transformDesc)
        checkCublasStatus(cublasLtMatrixTransformDescDestroy(transformDesc));

    // wait until device is done before freeing transformed buffers
    checkCudaStatus(cudaDeviceSynchronize());
    if (Ctransform)
        checkCudaStatus(cudaFree(Ctransform));
    if (Btransform)
        checkCudaStatus(cudaFree(Btransform));
    if (Atransform)
        checkCudaStatus(cudaFree(Atransform));
}