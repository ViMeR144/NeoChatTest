/*
 * CUDA kernels header file for neural network acceleration
 * Provides C++ interface for CUDA operations
 */

#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>

#ifdef __cplusplus
extern "C" {
#endif

// CUDA Neural Network class
typedef struct CudaNeuralNetwork CudaNeuralNetwork;

// Constructor and destructor
CudaNeuralNetwork* create_cuda_network();
void destroy_cuda_network(CudaNeuralNetwork* network);

// Matrix operations
void cuda_gemm(CudaNeuralNetwork* network, float* A, float* B, float* C, int m, int n, int k);

// Attention operations
void cuda_attention(CudaNeuralNetwork* network, float* Q, float* K, float* V, float* O,
                   int batch_size, int num_heads, int seq_len, int head_dim);

// Layer normalization
void cuda_layer_norm(CudaNeuralNetwork* network, float* input, float* output, 
                    float* gamma, float* beta, int batch_size, int seq_len, int hidden_size);

// Activation functions
void cuda_gelu(CudaNeuralNetwork* network, float* input, float* output, int size);

// Synchronization
void cuda_synchronize(CudaNeuralNetwork* network);

// Error checking macros
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error at %s:%d - %d\n", __FILE__, __LINE__, status); \
            exit(1); \
        } \
    } while(0)

#define CUDNN_CHECK(call) \
    do { \
        cudnnStatus_t status = call; \
        if (status != CUDNN_STATUS_SUCCESS) { \
            fprintf(stderr, "cuDNN error at %s:%d - %d\n", __FILE__, __LINE__, status); \
            exit(1); \
        } \
    } while(0)

#ifdef __cplusplus
}
#endif

#endif // CUDA_KERNELS_H

