#include <vector>
#include <omp.h>
#if defined(__ARM_NEON) || defined(__aarch64__)
    // ARM devices uses the dedicated SIMD kernels for higher speedup
    #include "libpopcntARM.h"
#else 
#ifdef GCC
    // GCC on Linux uses the nmmintrin.h
    #include <nmmintrin.h>
	#include <immintrin.h>
    #include <cstdint>
    #define popcnt64(a)       __builtin_popcountll(a)
#else
    // MSVC on Windows uses the intrin.h
    #include <intrin.h>	
    #define popcnt64(a)       __popcnt64(a)
#endif //GCC
#endif //ARM_NEON


// The bits of the container integer: int64_t
#define cntbits 64
// The bit width of quantized input values
#define BITS 2


template <typename T>
std::vector<T> Parallel_Img2Row_NHWCB_3x3(T* x, int N, int C, int H, int W, int KH, int KW, int stride1, int stride2) {

    const int OH = (H - KH + 1) / stride1;
    const int OW = (W - KW + 1) / stride2;
    const int H1 = OH * OW;
    const int W1 = KH * KW * C;
    std::vector<T> y = std::vector<T>(N * H1 * W1);

#pragma omp parallel for  
    for (int n = 0; n < N; n++) {
        int oh = 0;
        for (int h = 0; h < H - KH + 1; h += stride1) {
            int ow = 0;
            for (int w = 0; w < W - KW + 1; w += stride2) { 
                // y[((n * H1 + oh * OW + ow) * W1 + kh * KW * C + kwc)] = x[(((n * H + h + kh) * W + w) * C + kwc];
                T* y1 = &y[(n * H1 + oh * OW + ow) * W1];
                T* y2 = y1 + KW * C;
                T* y3 = y2 + KW * C;

                T* x1 = x + ((n * H + h) * W + w) * C;
                T* x2 = x1 + W * C;
                T* x3 = x2 + W * C;
#pragma omp simd
                for (int kwc = 0; kwc < KW * C; kwc++) {
                     y1[kwc] = x1[kwc];
                     y2[kwc] = x2[kwc];
                     y3[kwc] = x3[kwc];
                }
                ow++;
            }
            oh++;
        }
    }

    return y;
}


template <typename T>
std::vector<T> Parallel_Img2Row_NHWCB_5x5(T* x, int N, int C, int H, int W, int KH, int KW, int stride1, int stride2) {

    const int OH = (H - KH + 1) / stride1;
    const int OW = (W - KW + 1) / stride2;
    const int H1 = OH * OW;
    const int W1 = KH * KW * C;
    std::vector<T> y = std::vector<T>(N * H1 * W1);

#pragma omp parallel for  
    for (int n = 0; n < N; n++) {
        int oh = 0;
        for (int h = 0; h < H - KH + 1; h += stride1) {
            int ow = 0;
            for (int w = 0; w < W - KW + 1; w += stride2) { 
                // y[((n * H1 + oh * OW + ow) * W1 + kh * KW * C + kwc)] = x[(((n * H + h + kh) * W + w) * C + kwc];
                T* y1 = &y[(n * H1 + oh * OW + ow) * W1];
                T* y2 = y1 + KW * C;
                T* y3 = y2 + KW * C;
                T* y4 = y3 + KW * C;
                T* y5 = y4 + KW * C;

                T* x1 = x + ((n * H + h) * W + w) * C;
                T* x2 = x1 + W * C;
                T* x3 = x2 + W * C;
                T* x4 = x3 + W * C;
                T* x5 = x4 + W * C;
#pragma omp simd
                for (int kwc = 0; kwc < KW * C; kwc++) {
                     y1[kwc] = x1[kwc];
                     y2[kwc] = x2[kwc];
                     y3[kwc] = x3[kwc];
                     y4[kwc] = x4[kwc];
                     y5[kwc] = x5[kwc];
                }
                ow++;
            }
            oh++;
        }
    }

    return y;
}


template <typename T>
std::vector<T> Parallel_Img2Row_NHWCB_7x7(T* x, int N, int C, int H, int W, int KH, int KW, int stride1, int stride2) {

    const int OH = (H - KH + 1) / stride1;
    const int OW = (W - KW + 1) / stride2;
    const int H1 = OH * OW;
    const int W1 = KH * KW * C;
    std::vector<T> y = std::vector<T>(N * H1 * W1);

#pragma omp parallel for  
    for (int n = 0; n < N; n++) {
        int oh = 0;
        for (int h = 0; h < H - KH + 1; h += stride1) {
            int ow = 0;
            for (int w = 0; w < W - KW + 1; w += stride2) { 
                // y[((n * H1 + oh * OW + ow) * W1 + kh * KW * C + kwc)] = x[(((n * H + h + kh) * W + w) * C + kwc];
                T* y1 = &y[(n * H1 + oh * OW + ow) * W1];
                T* y2 = y1 + KW * C;
                T* y3 = y2 + KW * C;
                T* y4 = y3 + KW * C;
                T* y5 = y4 + KW * C;
                T* y6 = y5 + KW * C;
                T* y7 = y6 + KW * C;

                T* x1 = x + ((n * H + h) * W + w) * C;
                T* x2 = x1 + W * C;
                T* x3 = x2 + W * C;
                T* x4 = x3 + W * C;
                T* x5 = x4 + W * C;
                T* x6 = x5 + W * C;
                T* x7 = x6 + W * C;
#pragma omp simd
                for (int kwc = 0; kwc < KW * C; kwc++) {
                     y1[kwc] = x1[kwc];
                     y2[kwc] = x2[kwc];
                     y3[kwc] = x3[kwc];
                     y4[kwc] = x4[kwc];
                     y5[kwc] = x5[kwc];
                     y4[kwc] = x4[kwc];
                     y5[kwc] = x5[kwc];
                     y6[kwc] = x6[kwc];
                     y7[kwc] = x7[kwc];
                }
                ow++;
            }
            oh++;
        }
    }

    return y;
}
