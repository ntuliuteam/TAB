#include"TAB_GEMM.cuh"

__device__ void GetIdx(int* begin, int* end, const int x, const int size, const int X)
{
    if ((size < 1) || (X < 1) || (x >= size) || (x >= X)) {
        *begin = 0;
        *end = 0;
    }
    else {
        int remain = X % size;
        int unit = X / size;
        if (x < remain) {
            *begin = (unit + 1) * x;
            *end = (unit + 1) * (x + 1);
        }
        else if (remain < X) {
            *begin = remain + unit * x;
            *end = remain + unit * (x + 1);
        }
    }
}



__global__ void TNNGEMM(const int64_t* a, const int64_t* b, float* c, int M, int N, int K)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    const int MSIZE = blockDim.x * gridDim.x;
    const int NSIZE = blockDim.y * gridDim.y;

    int m_begin, m_end, n_begin, n_end;
    m_begin = m_end = n_begin = n_end = 0;

    GetIdx(&m_begin, &m_end, x, MSIZE, M);
    GetIdx(&n_begin, &n_end, y, NSIZE, N);
    const int KB = K * BITS;

    // It will do nothing if m_end==0 or n_end==0
    for (int m = m_begin; m < m_end; m++) {
        for (int n = n_begin; n < n_end; n++) {
            int cntp1 = 0;
            int cntp2 = 0;
            #pragma omp simd
            for (int k = 0; k < KB; k = k + BITS) {
                int64_t p1 = a[m * KB + k + 0] ^ b[n * KB + k + 0];
                int64_t p2 = a[m * KB + k + 1] & b[n * KB + k + 1];
                cntp1 = cntp1 + popcnt64(p2);
                cntp2 = cntp2 + popcnt64(p1 & p2);
            }
            c[m * N + n] = cntp1 - cntp2 - cntp2;
        }
    }
}


__global__ void TBNGEMM(const int64_t* a, const int64_t* b, float* c, int M, int N, int K)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    const int MSIZE = blockDim.x * gridDim.x;
    const int NSIZE = blockDim.y * gridDim.y;

    int m_begin, m_end, n_begin, n_end;
    m_begin = m_end = n_begin = n_end = 0;

    GetIdx(&m_begin, &m_end, x, MSIZE, M);
    GetIdx(&n_begin, &n_end, y, NSIZE, N);

    // It will do nothing if m_end==0 or n_end==0
    for (int m = m_begin; m < m_end; m++) {
        for (int n = n_begin; n < n_end; n++) {
            int cntp1 = 0;
            int cntp2 = 0;
            #pragma omp simd
            for (int k = 0; k < K; k++) {
                int64_t p1 = a[(m * K + k) * BITS + 0] ^ b[n * K + k];
                int64_t p2 = a[(m * K + k) * BITS + 1];
                cntp1 = cntp1 + popcnt64(p2);
                cntp2 = cntp2 + popcnt64(p2 & p1);
            }
            c[m * N + n] = cntp1 - cntp2 - cntp2;
        }
    }
}


__global__ void BTNGEMM(const int64_t* a, const int64_t* b, float* c, const int* btn, int M, int N, int K)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    const int MSIZE = blockDim.x * gridDim.x;
    const int NSIZE = blockDim.y * gridDim.y;

    int m_begin, m_end, n_begin, n_end;
    m_begin = m_end = n_begin = n_end = 0;

    GetIdx(&m_begin, &m_end, x, MSIZE, M);
    GetIdx(&n_begin, &n_end, y, NSIZE, N);

    // It will do nothing if m_end==0 or n_end==0
    for (int m = m_begin; m < m_end; m++) {
        for (int n = n_begin; n < n_end; n++) {
            int cntp2 = 0;
            #pragma omp simd
            for (int k = 0; k < K; k++) {
                int64_t p1 = a[m * K + k] ^ b[(n * K + k) * BITS + 0];
                cntp2 = cntp2 + popcnt64(p1 & b[(n * K + k) * BITS + 1]);
            }
            c[m * N + n] = btn[n] - cntp2 - cntp2;
        }
    }
}


__global__ void BNNGEMM(const int64_t* a, const int64_t* b, float* c, int M, int N, int K, int NUM)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    const int MSIZE = blockDim.x * gridDim.x;
    const int NSIZE = blockDim.y * gridDim.y;

    int m_begin, m_end, n_begin, n_end;
    m_begin = m_end = n_begin = n_end = 0;

    GetIdx(&m_begin, &m_end, x, MSIZE, M);
    GetIdx(&n_begin, &n_end, y, NSIZE, N);

    // It will do nothing if m_end==0 or n_end==0
    for (int m = m_begin; m < m_end; m++) {
        for (int n = n_begin; n < n_end; n++) {
            int cnt = 0;
            #pragma omp simd
            for (int k = 0; k < K; k++) {
                cnt = cnt + popcnt64(a[m * K + k] ^ b[n * K + k]);
            }
            c[m * N + n] = NUM - cnt - cnt;
        }
    }
}


__global__ void BTN_Weight(const int64_t* a, int* c, int N, int K)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int KB=K*BITS;
    int cnt = 0;
    const int64_t * a_base = a + x * KB + 1;
    for (int k = 0; k < KB; k=k+BITS) {
        cnt = cnt + popcnt64(a_base[k]);
    }
    c[x] = cnt;
}


__global__ void TerOnlyKernel(int64_t* t, const float* a, const float* ths, const int64_t* onebit, const int N, const  int H, const int W, const int C)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int bx = blockIdx.x;
    int begin = 0;
    int end = 0;

    GetIdx(&begin, &end, x, blockDim.x * gridDim.x, N * H * W);

    int PC = C / CNTBITS;
    if ((C % CNTBITS) > 0) {
        PC = PC + 1;
    }

    for (int nhw = begin; nhw < end; nhw++) {
        int pc = 0;
        int i = 0;
        int64_t bit1 = 0;
        int64_t bit2 = 0;
        #pragma omp simd
        for (int c = 0; c < C; c++) {
            if (a[nhw * C + c] > ths[bx]) {
                // quantize as +1
                bit2 = bit2 | onebit[i];
            }
            else if (a[nhw * C + c] < -ths[bx]) {
                // quantize as -1
                bit1 = bit1 | onebit[i];
                bit2 = bit2 | onebit[i];
            }
            i++;
            if (i == CNTBITS) {
                i = 0;
                t[(nhw * PC + pc) * BITS + 0] = bit1;
                t[(nhw * PC + pc) * BITS + 1] = bit2;
                bit1 = 0;
                bit2 = 0;
                pc = pc + 1;
            }
        }
        if (C % CNTBITS > 0) {
            t[(nhw * PC + pc) * BITS + 0] = bit1;
            t[(nhw * PC + pc) * BITS + 1] = bit2;
        }
    }
}


__global__ void BinOnlyKernel(int64_t* t, const float* a, const float* ths, const int64_t* onebit, const int N, const  int H, const int W, const int C)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int bx = blockIdx.x;
    int begin = 0;
    int end = 0;

    GetIdx(&begin, &end, x, blockDim.x * gridDim.x, N * H * W);

    int PC = C / CNTBITS;
    if ((C % CNTBITS) > 0) {
        PC = PC + 1;
    }

    for (int nhw = begin; nhw < end; nhw++) {
        int pc = 0;
        int i = 0;
        int64_t bit1 = 0;
        #pragma omp simd
        for (int c = 0; c < C; c++) {
            if (a[nhw * C + c] < ths[bx]) {
                // quantize as -1
                bit1 = bit1 | onebit[i];
            }
            i++;
            if (i == CNTBITS) {
                i = 0;
                t[nhw * PC + pc] = bit1;
                bit1 = 0;
                pc = pc + 1;
            }
        }
        if (C % CNTBITS > 0) {
            t[nhw * PC + pc] = bit1;
        }
    }
}


std::vector<torch::Tensor> QuanWithCuda_TAB(torch::Tensor X, torch::Tensor thresholds, int bitwidth, int N, int H, int W, int C)
{
    int64_t one = 1;
    int64_t bit64[CNTBITS];
    // 64-bits, set each bit
    for (int i = 0; i < CNTBITS; i++) {
        bit64[i] = one;
        one = one << 1;
    }

    // type
    // 0: TNN, 1: TBN, 2: BTN, 3: BNN 
    int PC = C / CNTBITS;
    if (C % CNTBITS) {
        PC = PC + 1;
    }
    int MemC, bits;
    if (bitwidth==1){
        // BTN, BNN use 1-bit in the activation
        MemC = PC;
        bits=1;
    }
    else{
        // TNN, TBN use 2-bit
        MemC = PC * BITS;
        bits=2;
    }
        

    float* dev_a = (float*)X.toType(torch::kF32).data_ptr();
    float* ths_a = (float*)thresholds.toType(torch::kF32).data_ptr();
    auto QX = torch::zeros(N*H*W*MemC, torch::device(torch::kCUDA).dtype(torch::kInt64));
    int64_t* img_a=(int64_t*)QX.data_ptr();
    auto BTN = torch::zeros(N, torch::device(torch::kCUDA).dtype(torch::kInt32));
    int * btn=(int *)BTN.data_ptr();
    const int img_a_size = N*H*W*MemC;
    const int btn_size = N;
    int64_t *onebit;
    cudaError_t cudaStatus;

    // Allocate Memory
    {
        cudaStatus = cudaMalloc((void**)&onebit, CNTBITS * sizeof(int64_t));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc onebit failed!");
            goto Error;
        }

        cudaStatus = cudaMemcpy(onebit, bit64, CNTBITS * sizeof(int64_t), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy onebit failed!");
            goto Error;
        }
    }


    // Quantize
    // type
    // 0: TNN, 1: TBN, 2: BTN, 3: BNN 
    if (bitwidth>1) { 
        // TerOnlyKernel(int64_t* t, const float* a, const float* ths, const int64_t* onebit, const int N, const  int H, const int W, const int C)
        TerOnlyKernel << <N * H, W >> > (img_a, dev_a, ths_a, onebit, N, H, W, C);
        BTN_Weight <<<16, N/16>>>(img_a, btn, N, H*W*PC);
    }
    else {
        // BinOnlyKernel(int64_t* t, const float* a, const float* ths, const int64_t* onebit, const int N, const  int H, const int W, const int C)
        BinOnlyKernel << <N * H, W >> > (img_a, dev_a, ths_a, onebit, N, H, W, C);
    }
    
Error:
    cudaFree(onebit);
    return {QX.reshape({ N, H, W, PC, bits}), BTN};
}


torch::Tensor ConvWithCuda_TAB(torch::Tensor X, torch::Tensor QW, torch::Tensor thresholds, torch::Tensor btn, int type,  int padding1, int padding2, int stride1, int stride2,  int N, int H, int W, int C, int KN, int KH, int KW)
{
	const int packH = H + 2 * padding1;
	const int packW = W + 2 * padding2;
	const int FH = (int)((packH - KH + 1) / stride1);
	const int FW = (int)((packW - KW + 1) / stride2);
	std::vector<float> y = std::vector<float>(N * KN * FH * FW);

	int64_t one = 1;
	int64_t bit64[CNTBITS];
	// 64-bits, set each bit
	for (int i = 0; i < CNTBITS; i++) {
		bit64[i] = one;
		one = one << 1;
	}

    float* dev_a = (float*)X.toType(torch::kF32).data_ptr();
    float* ths_a = (float*)thresholds.toType(torch::kF32).data_ptr();
    int64_t* pad_a = 0;
    int64_t* img_a = 0;
    int64_t* loc_a = 0;
    int64_t* dev_b =  (int64_t*)QW.toType(torch::kInt64).data_ptr();
    int* btn_b = (int *)btn.toType(torch::kInt32).data_ptr();
    
    auto Y = torch::zeros(N*KN*FH*FW, torch::device(torch::kCUDA));
    float* dev_c = (float*)Y.data_ptr();

    int64_t* onebit = 0;
    cudaError_t cudaStatus;
    const int PH = H + 2 * padding1;
    const int PW = W + 2 * padding2;
    const int OH = (PH - KH + 1) / stride1;
    const int OW = (PW - KW + 1) / stride2;

    // type
    // 0: TNN, 1: TBN, 2: BTN, 3: BNN 
    int PC = C / CNTBITS;
    if (C % CNTBITS) {
        PC = PC + 1;
    }
    int MemKC, MemC;
{
    if ((type == 2) || (type == 3))
        // BTN, BNN use 1-bit in the activation
        MemC = PC;
    else
        // TNN, TBN use 2-bit
        MemC = PC * BITS;

    if ((type == 1) || (type == 3))
        // TBN and BNN use 1-bit in weights
        MemKC = PC;
    else
        // TNN, BTN use 2-bit
        MemKC = PC * BITS;
}

    


    const int pad_a_size = N * PH * PW * MemC;
    const int img_a_size = N * PH * PW * KH * KW * MemC;
    const int dev_c_size = N * KN * OH * OW;

    // Allocate Memory
    {
        cudaStatus = cudaMalloc((void**)&pad_a, pad_a_size * sizeof(int64_t));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc pad_a failed!");
            goto Error;
        }
    
        cudaStatus = cudaMalloc((void**)&img_a, img_a_size * sizeof(int64_t));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc img_a failed!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&onebit, CNTBITS * sizeof(int64_t));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc onebit failed!");
            goto Error;
        }

        if (bit64 != NULL) {
            cudaStatus = cudaMemcpy(onebit, bit64, CNTBITS * sizeof(int64_t), cudaMemcpyHostToDevice);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMemcpy onebit failed!");
                goto Error;
            }
        }
    }

    // Quantize
    if ((type < 2)) { // TNN // TBN 
        // TerOnlyKernel(int64_t* t, const float* a, const float* ths, const int64_t* onebit, const int N, const  int H, const int W, const int C)
        TerOnlyKernel << <N * H, W >> > (img_a, dev_a, ths_a, onebit, N, H, W, C);
    }
    else { // BTN // BNN
        // BinOnlyKernel(int64_t* t, const float* a, const float* ths, const int64_t* onebit, const int N, const  int H, const int W, const int C)
        BinOnlyKernel << <N * H, W >> > (img_a, dev_a, ths_a, onebit, N, H, W, C);
    }


    // cuda check
    {
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Quantization launch failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }

        cudaStatus = cudaDeviceSynchronize();
        // Check for any errors launching the kernel
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Quantization!\n", cudaStatus);
            goto Error;
        }
    }


    // Do Padding when necessary
    bool NeedImg2Row = (stride1> 1) || (stride2 > 1) || (KH > 1) || (KW > 1);
    if ((padding1 > 0) || (padding2 > 0)) {
        PadKernel << <N, H >> > (pad_a, img_a, N, H, W, MemC, padding1, padding2);
        loc_a = pad_a;
    }
    else {
        if (NeedImg2Row) {
            cudaStatus = cudaMemcpy(pad_a, img_a, pad_a_size * sizeof(int64_t), cudaMemcpyDeviceToDevice);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMemcpy dev_a failed!");
                goto Error;
            }
            loc_a = pad_a;
        }
        else
            loc_a = img_a;
    }

    // cuda check
    {
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Padding launch failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }

        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Padding!\n", cudaStatus);
            goto Error;
        }
    }


    // Do Img2Row when ne3cessary
    if (NeedImg2Row) {
        // Img2Row
        dim3 dimImgGrid(N, OH);
        // Img2RowKernel(T* y, const T* a, const int N, const int H, const int W, const int C, const int KH, const int KW, const int S1, const int S2)
        Img2RowKernel << < dimImgGrid, OW >> > (img_a, loc_a, N, PH, PW, MemC, KH, KW, stride1, stride2);
        loc_a = img_a;
    }

    // cuda check
    {
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Img2Row launch failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }

        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Img2Row!\n", cudaStatus);
            goto Error;
        }
    }



    // GEMM to execute the real convolution and fully connected layers
    //GEMMWithCuda(const T * a, const T * b, float* c, const int* btn, const int M, const int N, const int K, const int NUM, const int type)
    cudaStatus = GEMMWithCuda(loc_a, dev_b, dev_c, btn_b, N * OH * OW, KN, KH * KW * PC, KH * KW * C, type);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "GENMM failed!");
        goto Error;
    }
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching GEMM!\n", cudaStatus);
        goto Error;
    }


Error:
    cudaFree(pad_a);
    cudaFree(img_a);
    cudaFree(onebit);

    return Y.reshape({ N, KN, FH, FW });
}
