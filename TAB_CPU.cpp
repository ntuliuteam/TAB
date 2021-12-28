// Please define GCC when using GCC on GNU/Linux based OS
//#define GCC
// Please define __ARM_NEON on ARM based devices, e.g., Raspberry Pi
//#define  __ARM_NEON


#include <torch/extension.h>
#include "TAB_CPU.h"


// Quantize the input x to be {+1, 0, -1} 
// Input:
//   x: the data to be quantized, using N_C_H_W data format
//   padding1: the padding around Height
//   padding2: the padding around Width
//   ths: the threshold values of each filter or input image or activation
//   N: batch size or filter number, C: Channel, H: Height, W: Width
// Output:
//   qx: the quantized x, using N, H, W, C, B format
std::vector<int64_t> Parallel_Ternarize_NCHWB_to_NHWCB(float* x, int padding1, int padding2, float* ths, int N, int C, int H, int W) {
    const int64_t one = 1;
    int64_t onebit[cntbits];
    // 64-bits, set each bit
    for (int i = 0; i < cntbits; i++) {
        onebit[i] = one << i;
    }

    // initial packed channel num
    const int priChannel = C / cntbits;
    // packC: actual packed input channel
    const int packC = (C % cntbits) ? (priChannel + 1) : priChannel;
    const int packH = H + 2 * padding1;
    const int packW = W + 2 * padding2;
    const int packWB = packW * BITS;
    const int HW = H * W;
    const int cntHW = cntbits * HW;
    // The PyTorch data always uses N, C, H, W format, no matter how we permute the data
    // torch::Tensor qx = torch::zeros({ N, packH, packW, packC }, torch::dtype(torch::kInt64));
    std::vector<int64_t> qx = std::vector<int64_t>(N * packC * packH * packW * BITS, 0);
    int64_t* qxptr = qx.data();



#pragma omp parallel for
    for (int in = 0; in < N; in++) {
        for (int ih = 0; ih < H; ih++) {
            int ow = padding2 * BITS;              
            for (int iw = 0; iw < W; iw++) {
                 // x[((in * C + (ic * cntbits + bit)) * H + ih) * W + iw]
                const float * x_base = x + (in * C * H + ih) * W + iw;
                //  qxptr[((in * packC + ic) * packH + (ih + padding1)) * packWB + ow + 0] = p1;
                int64_t* qx_base = qxptr + ((in  * packH + (ih + padding1)) * packW + (iw + padding2)) * packC * BITS;

                // Pack the first part: 0 ~ priChannel*cntbits
                for (int ic = 0; ic < priChannel; ic++) {
                    // for 2-bit packing
                    int64_t p1 = 0;
                    int64_t p2 = 0;
                    x_base+=cntHW;      

#pragma omp simd
                    for (int bit = 0; bit < cntbits; bit++) {
                        // PyTorch uses N_C_H_W format
                        // x.index({in, ic*cntbits+bit, ih, iw})
                        // float currentx = x[((in * C + (ic * cntbits + bit)) * H + ih) * W + iw];
                        float currentx = x_base[bit * HW];
                        if (currentx > ths[in]) {
                            // Pack 1: 01

                            p2 = p2 | onebit[bit];
                        }
                        else if (currentx < (-ths[in])) {
                            // Pack -1: 11
                            p1 = p1 | onebit[bit];
                            p2 = p2 | onebit[bit];
                        }
                    }
                    // Store the ternarized and packed data in N_C_H_W_B format
                    //qx.index({ in, ih + padding1, iw + padding2, priChannel * 2 + 0 }) = p1;
                    //qx.index({ in, ih + padding1, iw + padding2, priChannel * 2 + 1 }) = p2;
                    // qxptr[((in * packC + ic) * packH + (ih + padding1)) * packWB + ow + 0] = p1;
                    // qxptr[((in * packC + ic) * packH + (ih + padding1)) * packWB + ow + 1] = p2;
                    qx_base[0]=p1;
                    qx_base++;
                    qx_base[0]=p2;
                    qx_base++;
                }
                if ((C % cntbits) > 0) {
                    // Pack the second part: priChannel*cntbits ~ C
                    int64_t p1 = 0;
                    int64_t p2 = 0;
                    x_base+=cntHW; 
#pragma omp simd
                    for (int bit = 0; bit < (C % cntbits); bit++) {
                        // float currentx = x[((in * C + (priChannel * cntbits + bit)) * H + ih) * W + iw];
                         float currentx = x_base[bit * HW];
                        if (currentx > ths[in]) {
                            // Pack 1: 01

                            p2 = p2 | onebit[bit];
                        }
                        else if (currentx < (-ths[in])) {
                            // Pack -1: 11
                            p1 = p1 | onebit[bit];
                            p2 = p2 | onebit[bit];
                        }
                    }
                    //qxptr[((in * packC + priChannel) * packH + (ih + padding1)) * packWB + ow + 0] = p1;
                    //qxptr[((in * packC + priChannel) * packH + (ih + padding1)) * packWB + ow + 1] = p2;
                    qx_base[0]=p1;
                    qx_base++;
                    qx_base[0]=p2;
                    qx_base++;
                }
                // Update ow
                ow += BITS;
            }
        }

    }


    return qx;
}



// Quantize the input x to be {+1, -1} 
// Input:
//   x: the data to be quantized, using N_C_H_W data format
//   padding1: the padding around Height
//   padding2: the padding around Width
//   ths: the threshold values of each filter or input image or activation
//   N: batch size or filter number, C: Channel, H: Height, W: Width
// Output:
//   qx: the quantized x, using N, H, W, C format
std::vector<int64_t> Parallel_Binarize_NCHW_to_NHWC(const float* x, int padding1, int padding2, int N, int C, int H, int W) {
    const int64_t one = 1;
    int64_t onebit[cntbits];
    // 64-bits, set each bit
    for (int i = 0; i < cntbits; i++) {
        onebit[i] = one << i;
    }

    // initial packed channel num
    const int priChannel = C / cntbits;
    // packC: actual packed input channel
    const int packC = (C % cntbits) ? (priChannel + 1) : priChannel;
    const int packH = H + 2 * padding1;
    const int packW = W + 2 * padding2;
    const int HW = H * W;
    const int cntHW = cntbits * HW;
    // The PyTorch data always uses N, C, H, W format, no matter how we permute the data
    // torch::Tensor qx = torch::zeros({ N, packH, packW, packC }, torch::dtype(torch::kInt64));
    std::vector<int64_t> qx = std::vector<int64_t>(N * packC * packH * packW, 0);
    int64_t* qxptr = qx.data();

   
#pragma omp parallel for
    for (int in = 0; in < N; in++) {
        for (int ih = 0; ih < H; ih++) {
            for (int iw = 0; iw < W; iw++) {
                // x[((in * C + (ic * cntbits + bit)) * H + ih) * W + iw]
                const float * x_base = x + (in * C * H + ih) * W + iw;
                // qxptr[((in * packC + ic) * packH + (ih + padding1)) * packW + (iw + padding2)] 
                int64_t* qx_base = qxptr + ((in  * packH + (ih + padding1)) * packW + (iw + padding2)) * packC;
                // Pack the first part: 0 ~ priChannel*cntbits
                for (int ic = 0; ic < priChannel; ic++) {
                    // for 1-bit packing
                    int64_t p1 = 0;
                    x_base+=cntHW;
#pragma omp simd
                    for (int bit = 0; bit < cntbits; bit++) {
                        // PyTorch uses N_C_H_W format
                        // x.index({in, ic*cntbits+bit, ih, iw})
                        // if (x[((in * C + (ic * cntbits + bit)) * H + ih) * W + iw] < 0) {
                        if (x_base[bit * HW] < 0) {
                            // Pack -1: 1
                            p1 = p1 | onebit[bit];
                        }
                    }
                    // Store the binarized and packed data in N_C_H_W format
                    // qxptr[((in * packC + ic) * packH + (ih + padding1)) * packW + (iw + padding2)] = p1;
                    qx_base[ic]=p1;
                }
                if ((C % cntbits) > 0) {
                    // Pack the second part: priChannel*cntbits ~ C
                    int64_t p1 = 0;
                    x_base+=cntHW;      
#pragma omp simd
                    for (int bit = 0; bit < (C % cntbits); bit++) {
                        // if (x[((in * C + (priChannel * cntbits + bit)) * H + ih) * W + iw] < 0) {
                        if (x[bit * HW] < 0) {
                            // Pack -1: 1
                            p1 = p1 | onebit[bit];
                        }
                    }
                    // qxptr[((in * packC + priChannel) * packH + (ih + padding1)) * packW + (iw + padding2)] = p1;
                    qx_base[priChannel]=p1;
                }
            }
        }
    }
   
    return qx;
}



// The wraper function that calls Ternarization functions.
// Deal with the upper PyTorch tensors and provide unified APIs to upper Python code 
std::vector<torch::Tensor> TAB_Quantize(torch::Tensor X, torch::Tensor thresholds, int bitwidth, int N, int H, int W, int C){
	float* x = (float*)input.toType(torch::kF32).data_ptr();
    float* ths = (float*)thresholds.toType(torch::kF32).data_ptr();
    auto BTN = torch::zeros(N, torch::dtype(torch::kInt32));
    int * btn=(int *)BTN.data_ptr();

	// initial packed channel num
    const int priChannel = C / cntbits;
    // packC: actual packed input channel
    const int packC = (C % cntbits) ? (priChannel + 1) : priChannel;
    const int packH = H + 2 * padding1;
    const int packW = W + 2 * padding2;

    std::vector<int64_t> qx;
    if (bitwidth==1)
        qx = Parallel_Binarize_NCHW_to_NHWC(x, padding1, padding2, ths, N, C, H, W);
    else{

        qx = Parallel_Ternarize_NCHWB_to_NHWCB(x, padding1, padding2, ths, N, C, H, W);
    }

    return torch::tensor(qx, torch::dtype(torch::kInt64)).reshape({ N, packC, packH, packW, bitwidth });
}




// TABGEMM: TNN
// In M-K, N-K order, M-N, 
// K is the absolute K, it should *BITS to get the real memory boundary
std::vector<int> TNNGEMM_baseline(int64_t* a, int64_t* b, int M, int N, int K) {
    std::vector<int> y = std::vector<int>(M * N, 0);
    const int KB = K * BITS;

#pragma omp parallel for 
    for (int oh = 0; oh < M; oh++) {
        for (int ow = 0; ow < N; ow++) {
            int cntp1 = 0;
            int cntp2 = 0;
#pragma omp simd
            for (int iw = 0; iw < KB; iw += BITS) {
                // Use H_W_B format
                int64_t p1 = a[oh * KB + iw + 0] ^ b[ow * KB + iw + 0];
                int64_t p2 = a[oh * KB + iw + 1] & b[ow * KB + iw + 1];
                cntp1 = cntp1 + popcnt64(p2);
                cntp2 = cntp2 + popcnt64(p1 & p2);
            }
            y[oh * N + ow] = cntp1 - cntp2 - cntp2;
        }
    }
    return y;
}


// In M-K, N-K order, TBN, Ternary-Activation Binary-Weight
std::vector<int> TBNGEMM_baseline(int64_t* a, int64_t* b, int M, int N, int K) {
    std::vector<int> y = std::vector<int>(M * N);

#pragma omp parallel for  
    for (int oh = 0; oh < M; oh++) {
        for (int ow = 0; ow < N; ow++) {
            int cntp1 = 0;
            int cntp2 = 0;
#pragma omp simd
            for (int iw = 0; iw < K; iw++) {
                // Use H_W_B format
                int64_t p1 = a[(oh * K + iw) * BITS + 0] ^ b[ow * K + iw];
                int64_t p2 = a[(oh * K + iw) * BITS + 1];
                cntp1 = cntp1 + popcnt64(p2);
                cntp2 = cntp2 + popcnt64(p1 & p2);
            }
            y[oh * N + ow] = cntp1 - cntp2 - cntp2;
        }
    }
    return y;
}


// In M-K, N-K order, BTN, Binary-Activation Ternary-Weight
std::vector<int> BTNGEMM_baseline(int64_t* a, int64_t* b, int* cnt1, int M, int N, int K) {
    std::vector<int> y = std::vector<int>(M * N);

#pragma omp parallel for  
    for (int oh = 0; oh < M; oh++) {
        for (int ow = 0; ow < N; ow++) {
            int cntp2 = 0;
#pragma omp simd
            for (int iw = 0; iw < K; iw++) {
                // Use H_W_B format
                int64_t p1 = a[oh * K + iw] ^ b[(ow * K + iw) * BITS + 0];
                cntp2 = cntp2 + popcnt64(p1 & b[(ow * K + iw) * BITS + 1]);
            }
            y[oh * N + ow] = cnt1[ow] - cntp2 - cntp2;
        }
    }
    return y;
}


// In M-K, N-K order, BNN, Binary-Activation Binary-Weight
std::vector<int> BNNGEMM_baseline(int64_t* a, int64_t* b, int M, int N, int K, int NUM) {
    std::vector<int> y = std::vector<int>(M * N);

#pragma omp parallel for  
    for (int oh = 0; oh < M; oh++) {
        for (int ow = 0; ow < N; ow++) {
            int cntp1 = 0;
#pragma omp simd
            for (int iw = 0; iw < K; iw++) {
                // Use H_W_B format
                cntp1 = cntp1 + popcnt64(a[oh * K + iw] ^ b[ow * K + iw]);
            }
            y[oh * N + ow] = NUM - cntp1 - cntp1;
        }
    }
    return y;
}


// In M-K, N-K order, DoReFa-Net 2-bit 
std::vector<int> DRFGEMM_baseline(int64_t* a, int64_t* b, int M, int N, int K) {
    std::vector<int> y = std::vector<int>(M * N);

#pragma omp parallel for  
    for (int oh = 0; oh < M; oh++) {
        for (int ow = 0; ow < N; ow++) {
            int cntp1 = 0;
            int cntp2 = 0;
            int cntp3 = 0;
            int cntp4 = 0;
#pragma omp simd
            for (int iw = 0; iw < K; iw++) {
                // Use H_W_B format
                int64_t p1 = (a[(oh * K + iw) * BITS + 0] & b[(ow * K + iw) * BITS + 0]);
                int64_t p2 = (a[(oh * K + iw) * BITS + 1] & b[(ow * K + iw) * BITS + 0]);
                int64_t p3 = (a[(oh * K + iw) * BITS + 0] & b[(ow * K + iw) * BITS + 1]);
                int64_t p4 = (a[(oh * K + iw) * BITS + 1] & b[(ow * K + iw) * BITS + 1]);
                cntp1 = cntp1 + popcnt64(p1);
                cntp2 = cntp2 + popcnt64(p2);
                cntp3 = cntp3 + popcnt64(p3);
                cntp4 = cntp4 + popcnt64(p4);
            }
            y[oh * N + ow] = (cntp1 << 2) + ((cntp2 + cntp3) << 1) + cntp4;
        }
    }
    return y;
}


// In M-K, N-K order, RTN, Reparatermized Rernary Network 
std::vector<int> RTNGEMM_baseline(int64_t* a, int64_t* b, int M, int N, int K) {
    std::vector<int> y = std::vector<int>(M * N);
    const int64_t mask = 0x5555555555555555;

#pragma omp parallel for  
    for (int oh = 0; oh < M; oh++) {
        for (int ow = 0; ow < N; ow++) {
            int cntp1 = 0;
            int cntp2 = 0;
#pragma omp simd
            for (int iw = 0; iw < K; iw++) {
                // Use H_W_B format
                int64_t p1 = (a[oh * K + iw] ^ b[ow * K + iw]) >> 1;
                int64_t p2 = (a[oh * K + iw] & b[ow * K + iw]) & mask;
                cntp1 = cntp1 + popcnt64(p2);
                cntp2 = cntp2 + popcnt64(p1 & p2);
            }
            y[oh * N + ow] = cntp1 - cntp2 - cntp2;
        }
    }
    return y;
}




#if defined(__ARM_NEON) || defined(__aarch64__) // If it's ARM CPU

// In M-K, N-K order, M-N
// TNN GEMM using the TAB-TNN kernel in libpopcntARM
std::vector<int> TNNGEMM_Kernel(int64_t* a, int64_t* b, int M, int N, int K) {
    std::vector<int> y = std::vector<int>(M * N);
    const int KB = K * BITS;
#pragma omp parallel for  
    for (int oh = 0; oh < M; oh++) {
#pragma omp simd
        for (int ow = 0; ow < N; ow++) {            
            y[oh * N + ow] = TNNpopKernel(&a[oh * KB], &b[ow * KB], K);
        }
    }
    return y;
}


// In M-K, N-K order, TBN, Ternary-Activation Binary-Weight
std::vector<int> TBNGEMM_Kernel(int64_t* a, int64_t* b, int M, int N, int K) {
    std::vector<int> y = std::vector<int>(M * N);
    const int KB = K * BITS;
#pragma omp parallel for  
    for (int oh = 0; oh < M; oh++) {
#pragma omp simd 
        for (int ow = 0; ow < N; ow++) {
            y[oh * N + ow] = TBNpopKernel(&a[oh * KB], &b[ow * K], K);
        }
    }
    return y;
}


// In M-K, N-K order, BTN, Binary-Activation Ternary-Weight
std::vector<int> BTNGEMM_Kernel(int64_t* a, int64_t* b, int * cnt1, int M, int N, int K) {
    std::vector<int> y = std::vector<int>(M * N);
    const int KB = K * BITS;
#pragma omp parallel for  
    for (int oh = 0; oh < M; oh++) {
#pragma omp simd
        for (int ow = 0; ow < N; ow++) {
            int cntp2 = BTNpopKernel(&a[oh * K], &b[ow * KB], K);
            y[oh * N + ow] = cnt1[ow] - cntp2 - cntp2;
        }
    }
    return y;
}


// In M-K, N-K order, BNN, Binary-Activation Binary-Weight
std::vector<int> BNNGEMM_Kernel(int64_t* a, int64_t* b, int M, int N, int K, int NUM) {
    std::vector<int> y = std::vector<int>(M * N);

#pragma omp parallel for  
    for (int oh = 0; oh < M; oh++) {
#pragma omp simd
        for (int ow = 0; ow < N; ow++) {
            int cntp1 = BNNpopKernel(&a[oh * K], &b[ow * K], K);
            y[oh * N + ow] = NUM - cntp1 - cntp1;
        }
    }
    return y;
}


// In M-K, N-K order, DoReFa-Net 2-bit 
std::vector<int> DRFGEMM_Kernel(int64_t* a, int64_t* b, int M, int N, int K) {
    std::vector<int> y = std::vector<int>(M * N);
    const int KB = K * BITS;
#pragma omp parallel for  
    for (int oh = 0; oh < M; oh++) {
#pragma omp simd
        for (int ow = 0; ow < N; ow++) {
            y[oh * N + ow] = DRFpopKernel(&a[oh * KB], &b[ow * KB], K);
        }
    }
    return y;
}


// In M-K, N-K order, RTN, Reparatermized Rernary Network 
std::vector<int> RTNGEMM_Kernel(int64_t* a, int64_t* b, int M, int N, int K) {
    std::vector<int> y = std::vector<int>(M * N);
#pragma omp parallel for  
    for (int oh = 0; oh < M; oh++) {
#pragma omp simd
        for (int ow = 0; ow < N; ow++) { 
            y[oh * N + ow] = RTNpopKernel(&a[oh * K], &b[ow * K], K);
        }
    }
    return y;
}


#endif // If it's ARM CPU




/* Container function of quantization and convolution functions. Can be applied to conv and FC layers.
* Conv: 1X1 and 3X3 kernels. FC equals to 1x1 conv.
* type: 
* 0: TAB-TNN
* 1: TAB-TBN
* 2: TAB-BTN
* 3: TAB-BNN
* 4: DoReFa-Net 2-bit
* 5: RTN
* */
// Ternary and Binary Convolution using N, H, W, C, B format
// Input: 
//   qx: quantized activation
//   qw: quantized weights
//   stride1: the stride on Height
//   stride2: the stride on Width
//   N: batch number, C, channel, H: Height, W: Width
//   KN: number of filters/kernels, KH: Kernel Height, KW, Kernel Width 
// Output:
//   yi: convolution result
std::vector<int> TAB_Conv(float * ix, float * ths, int64_t * qw, int * btn, int type, int p1, int p2, int s1, int s2, int batch_size, int c, int h, int w,
    int kn, int kh, int kw) {
    int ph, pw, oh, ow, pc;
    ph = h + 2 * p1;
    pw = w + 2 * p2;
    oh = (ph - kh + 1) / s1;
    ow = (pw - kw + 1) / s2;
    
    std::vector<int64_t> qx;
    std::vector<int> yi;

    // Quantize
    if (type == 5) { // RTN
        int half_cnt = cntbits / 2;
        pc = (c % half_cnt) ? ((c / half_cnt) + 1) : (c / half_cnt);
        qx = Parallel_Ternarize_RTN(ix, p1, p2, ths, batch_size, c, h, w);
    }
    else {
        pc = (c % cntbits) ? ((c / cntbits) + 1) : (c / cntbits);
        if ((type == 4) || (type == 5)) {
            qx = Parallel_Binarize_NCHW(ix, p1, p2, batch_size, c, h, w);
        }
        else {
            qx = Parallel_Ternarize_NCHWB(ix, p1, p2, ths, batch_size, c, h, w);
            pw = pw * BITS;
        }
    }       
    
    // Img2Col
    switch (kh) {
        case 3: {           
            qx = Parallel_Img2Row_NHWCB_3x3(qx.data(), batch_size, pc, ph, pw, kh, kw, s1, s2);
            break;
        }
        case 5: {           
            qx = Parallel_Img2Row_NHWCB_3x3(qx.data(), batch_size, pc, ph, pw, kh, kw, s1, s2);
            break;
        }
        case 7: {           
            qx = Parallel_Img2Row_NHWCB_3x3(qx.data(), batch_size, pc, ph, pw, kh, kw, s1, s2);
            break;
        }
    }   

#if defined(__ARM_NEON) || defined(__aarch64__) // If it's ARM CPU

    switch (type) {
        case 0: {
            yi = TNNGEMM_Kernel(qx.data(), qw, batch_size * oh * ow, kn, pc * kh * kw);
            break;
        }
        case 1: {
            yi = TBNGEMM_Kernel(qx.data(), qw, batch_size * oh * ow, kn, pc * kh * kw);
            break;
        }
        case 2: {
            yi = BTNGEMM_Kernel(qx.data(), qw, btn, batch_size * oh * ow, kn, pc * kh * kw);
            break;
        }
        case 3: { 
            yi = BNNGEMM_Kernel(qx.data(), qw, batch_size * oh * ow, kn, pc * kh * kw, c * kh * kw);
            break;
        }
        case 4: { 
            yi = DRFGEMM_Kernel(qx.data(), qw, batch_size * oh * ow, kn, pc * kh * kw);
            break;
        }
        case 5: {
            yi = RTNGEMM_Kernel(qx.data(), qw, batch_size * oh * ow, kn, pc * kh * kw);
            break;
        }
    }

#else // If it's Intel or AMD CPU

        switch (type) {
        case 0: {
            yi = TNNGEMM_baseline(qx.data(), qw, batch_size * oh * ow, kn, pc * kh * kw);
            break;
        }
        case 1: {
            yi = TBNGEMM_baseline(qx.data(), qw, batch_size * oh * ow, kn, pc * kh * kw);
            break;
        }
        case 2: {
            yi = BTNGEMM_baseline(qx.data(), qw, btn, batch_size * oh * ow, kn, pc * kh * kw);
            break;
        }
        case 3: { 
            yi = BNNGEMM_baseline(qx.data(), qw, batch_size * oh * ow, kn, pc * kh * kw, c * kh * kw);
            break;
        }
        case 4: { 
            yi = DRFGEMM_baseline(qx.data(), qw, batch_size * oh * ow, kn, pc * kh * kw);
            break;
        }
        case 5: {
            yi = RTNGEMM_baseline(qx.data(), qw, batch_size * oh * ow, kn, pc * kh * kw);
            break;
        }
    }
    }
    

#endif // If it's ARM, Intel or AMD CPU

    return yi;
}



// The interface of all the convolution functions
// Ternareize X --> QX, then do convolution between QX and QW
// Deal with the upper PyTorch tensors and provide unified APIs to upper Python code 
torch::Tensor TAB_Conv2d(torch::Tensor X, torch::Tensor QW, torch::Tensor thresholds, torch::Tensor BTN, int type, int padding1, int padding2, int stride1, int stride2, int N, int C, int H, int W, int KN, int KH, int KW)
{
	// Get the pointers of all the input data
    float* x = (float*)X.toType(torch::kF32).data_ptr();
    int64_t* qw = (int64_t*)QW.toType(torch::kInt64).data_ptr();
    float* ths = (float*)thresholds.toType(torch::kF32).data_ptr();
    int * btn = (int *)BTN.toType(torch::kInt32).data_ptr();

    // std::vector<int> TAB_Conv(float * ix, float * ths, int64_t * qw, int * btn, int type, int p1, int p2, int s1, int s2, int batch_size, int c, int h, int w, int kn, int kh, int kw)
	std::vector<int> y = TAB_Conv(x, ths, qw, btn, type, padding1, padding2, stride1, stride2, N, C, H, W, KN, KH, KW);
	
	return torch::tensor(y).reshape({ N, KN, FH, FW });
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("TAB_Quantize", &TAB_Quantize, "TAB Quantize: the ternarization and binarization functions on N, C, H, W data");
  m.def("TAB_Conv2d", &TAB_Conv2d, "TAB Conv2d: quantization on the input x, then do convolution between QX and QW");
}