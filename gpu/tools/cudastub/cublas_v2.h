#pragma once

// Structural stand-in for cublas - see cuda_runtime.h for what the stubs
// are and are not for. The GemmEx signature mirrors the real one so a
// wrong argument count in launchers.cu is caught here.

#include "cuda_runtime.h"

typedef struct cublasContext* cublasHandle_t;
typedef enum { CUBLAS_STATUS_SUCCESS = 0 } cublasStatus_t;
typedef enum { CUBLAS_OP_N = 0, CUBLAS_OP_T = 1, CUBLAS_OP_C = 2 } cublasOperation_t;
typedef enum { CUDA_R_32F = 0, CUDA_R_16F = 2 } cudaDataType_t;
typedef enum { CUBLAS_COMPUTE_32F = 68 } cublasComputeType_t;
typedef enum { CUBLAS_GEMM_DEFAULT = -1 } cublasGemmAlgo_t;

cublasStatus_t cublasCreate(cublasHandle_t* handle);
cublasStatus_t cublasDestroy(cublasHandle_t handle);
cublasStatus_t cublasGemmEx(cublasHandle_t handle, cublasOperation_t transa,
                            cublasOperation_t transb, int m, int n, int k,
                            const void* alpha, const void* A,
                            cudaDataType_t Atype, int lda, const void* B,
                            cudaDataType_t Btype, int ldb, const void* beta,
                            void* C, cudaDataType_t Ctype, int ldc,
                            cublasComputeType_t computeType,
                            cublasGemmAlgo_t algo);
