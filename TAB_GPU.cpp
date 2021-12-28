#include <torch/extension.h>
// Reference: https://pytorch.org/tutorials/advanced/cpp_extension.html#writing-a-mixed-c-cuda-extension
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "TAB_GEMM.cuh"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


std::vector<torch::Tensor> QuanWithCuda_TAB(torch::Tensor X, torch::Tensor thresholds, int bitwidth, int N, int H, int W, int C);

torch::Tensor ConvWithCuda_TAB(torch::Tensor X, torch::Tensor QW, torch::Tensor thresholds, torch::Tensor btn, int type,  int padding1, int padding2, int stride1, int stride2,  int N, int H, int W, int C, int KN, int KH, int KW);




// The convolution function interface of TAB series TNN, TBN, BTN and BNN
// Input: 
//   X: input activation
//   QW: quantized weights
//   ths: the thredhols values of each activation. length = N = batch size
//   btn: the pre-calculated popcnt(W2) of BTN
//   type: show the conv type. 0: TNN, 1, TBN, 2: BTN, 3: BNN
//   padding1: the padding around Height
//   padding2: the padding around Width
//   stride1: the stride on Height
//   stride2: the stride on Width
//   N: batch size, C, channel, H: Height, W: Width
//   KN: number of filters/kernels, KH: Kernel Height, KW, Kernel Width 
// Output:
//   y: convolution result
torch::Tensor TAB_Conv2d(torch::Tensor X, torch::Tensor QW, torch::Tensor thresholds, torch::Tensor btn, int type, int padding1, int padding2, int stride1, int stride2, int N,  int H, int W, int C, int KN, int KH, int KW)
{
	CHECK_INPUT(X);
	CHECK_INPUT(QW);
	CHECK_INPUT(thresholds);
	CHECK_INPUT(btn);
	
	return ConvWithCuda_TAB(X, QW, thresholds, btn, type, padding1, padding2, stride1, stride2, N, H, W, C, KN, KH, KW);
}



// Quantize the input x to be {+1, 0 -1} or {+1, -1}
// Input:
//   x: the data to be quantized, using N_H_W_C data format
//   ths: the threshold values of each filter or each input image or each activation
//   N: batch size or filter number, C: Channel, H: Height, W: Width
// Output:
//   qx: the quantized x, using N, C, H, W, B format
std::vector<torch::Tensor> TAB_Quantize(torch::Tensor X, torch::Tensor thresholds, int bitwidth, int N, int H, int W, int C){
	CHECK_INPUT(X); 
	CHECK_INPUT(thresholds);

	return QuanWithCuda_TAB(X, thresholds, bitwidth, N, H, W, C);
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("Quantize", &TAB_Quantize, "Ternarization or binarization of the weights in NHWC format");
	m.def("Conv2d",   &TAB_Conv2d, "Conv2d of all TAB series CNNS");
}