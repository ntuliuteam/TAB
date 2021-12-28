#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
// Torch Libs
#include <torch/extension.h>
#include <vector>

// Define the constants and intrinsics
#define BITS 2
#define CNTBITS 64
#define popcnt64(a) __popcll(a) 


__device__ void GetIdx(int* begin, int* end, const int x, const int size, const int X);
__global__ void TNNGEMM(const int64_t* a, const int64_t* b, float* c, int M, int N, int K);
__global__ void TBNGEMM(const int64_t* a, const int64_t* b, float* c, int M, int N, int K);
__global__ void BTNGEMM(const int64_t* a, const int64_t* b, float* c, const int* btn, int M, int N, int K);
__global__ void BNNGEMM(const int64_t* a, const int64_t* b, float* c, int M, int N, int K, int NUM);


// Wraper function for GEMM functions
template<typename T>
cudaError_t GEMMWithCuda(const T* a, const T* b, float* c, const int* btn, const int M, const int N, const int K, const int NUM, const int type) {
    cudaError_t cudaStatus;

    dim3 dimGrid((M/16+1), 16);
    dim3 dimBlock(16, (N/16+1));

    // Launch a kernel on the GPU with one thread for each element.
    switch (type) {
    case 0: {
        TNNGEMM << <dimGrid, dimBlock >> > (a, b, c, M, N, K);
        break;
    }
    case 1: {
        TBNGEMM << <dimGrid, dimBlock >> > (a, b, c, M, N, K);
        break;
    }
    case 2: {
        BTNGEMM << <dimGrid, dimBlock >> > (a, b, c, btn, M, N, K);
        break;
    }
    case 3: {
        BNNGEMM << <dimGrid, dimBlock >> > (a, b, c, M, N, K, NUM);
        break;
    }
    }

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "TAB and DoReFa GEMM Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching TAB and DoReFa GEMM Kernel!\n", cudaStatus);
        goto Error;
    }

Error:
    return cudaStatus;
}


__global__ void TerOnlyKernel(int64_t* t, const float* a, const float* ths, const int64_t* onebit, const int N, const  int H, const int W, const int C);
__global__ void BinOnlyKernel(int64_t* t, const float* a, const float* ths, const int64_t* onebit, const int N, const  int H, const int W, const int C);


template<typename T>
__global__ void PadKernel(T* y, const T* a, const int N, const int H, const int W, const int C, const int P1, const int P2)
{
    const int n = blockIdx.x;
    const int h = threadIdx.x;

    const int PH = H + P1 * 2;
    const int PW = W + P2 * 2;

    T* ybase = y + ((n * PH + h + P1) * PW + P2) * C;
    const T* abase = a + (n * H + h) * W * C;

    for (int wc = 0; wc < W * C; wc++) {
        // y[((n * PH + h + P1) * PW + P2) * C + wc] = a[((n * H + h) * WC + wc];
        ybase[wc] = abase[wc];
    }
}


template<typename T>
__global__ void Img2RowKernel(T* y, const T* a, const int N, const int H, const int W, const int C, const int KH, const int KW, const int S1, const int S2)
{
    const int n = blockIdx.x;
    const int oh = blockIdx.y;
    const int ow = threadIdx.x;

    const int OH = (H - KH + 1) / S1;
    const int OW = (W - KW + 1) / S2;
    const int H1 = OH * OW;
    const int W1 = C * KH * KW;
    const int KWC = KW * C;

    int h = oh * S1;
    int w = ow * S2;
    T* ybase = y + (n * H1 + oh * OW + ow) * W1;
    const T* abase = a + ((n * H + h) * W + w) * C;
    for (int kh = 0; kh < KH; kh++) {
        for (int kwc = 0; kwc < KWC; kwc++) {
            // y[(n * H1 + oh * OW + ow) * W1 + (kh * KW * C + kwc)] = a[(((n * H + h + kh) * W + w + kw) * C + kc)] = a[(((n * H + h) * W + w) * C + kh * W * C + kwc)];
            ybase[(kh * KWC + kwc)] = abase[kh * W * C + kwc];
        }
    }
}


// Helper function for using CUDA to Quant/Conv in parallel.
std::vector<torch::Tensor> QuanWithCuda_TAB(torch::Tensor X, torch::Tensor thresholds, int bitwidth, int N, int H, int W, int C);
torch::Tensor ConvWithCuda_TAB(torch::Tensor X, torch::Tensor QW, torch::Tensor thresholds, torch::Tensor btn, int type,  int padding1, int padding2, int stride1, int stride2,  int N, int H, int W, int C, int KN, int KH, int KW);
