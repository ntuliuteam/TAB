# TAB PyTorch Extension

TAB series Ternary And Binary convolution and quantization functions on ARM and x86 CPU and Nvidia GPU.

## File Organization
#### CUDA Version
 - Setup_GPU.py: the installation file of PyTorch Extension
 - TAB_GPU.cpp: the interface file that wraps the cuda functions
 - TAB_GEMM.cuh: the head file 
 - TAB_GEMM.cu:  the source file contains all the cuda functions
#### CPU Version
 - Setup_GCC.py: Setup on x86-x64 CPU with GCC 
 - Setup_MSVC.py: Setup on x86-x64 CPU with MSVC
 - Setup_ARM.py: Setup on ARM CPU with GCC
 - libpopcntARM.h: The headfile with dedicated popcnt kernels on ARM CPU
 - TAB_CPU.h: The haedfile of CPU functions
 - TAB_CPU.cpp: The CPU functions

## Install and demo
Taking the CUDA version as example
 - Run this command in conda: " python Setup_GPU.py install "
 - Try it out in the " TAB Demo GPU.ipynb "

## Working Configuration 
The CUDA and MSVC version 
 - Windows 10
 - CUDA 11.1 on RTX-3080
 - PyTorch 1.8.1 
 - Conda Python version: 3.8.5
 - MS Visual Studio Community 2019

 
The ARM version
 - Rpi 400 with Rasberry Pi OS
 - GCC 8.3.0

## API Details

 - TAB_Qunatize(torch::Tensor X, torch::Tensor thresholds, int bitwidth, int N, int H, int W, int C)
   - It returns two tensors: quantized QX, the pre-calculated popcnt(W2) of BTN: BTN_W

// Quantize the input x to be {+1, 0 -1} or {+1, -1}  
// Input:  
//   x: the data to be quantized, using N_H_W_C data format  
//   ths: the threshold values of each filter or each input image or each activation  
//   bit-width: determins whether this is Ternarization or Binarization  
//   N: batch size or filter number, C: Channel, H: Height, W: Width  
// Output:  
//   qx: the quantized x, using N, C, H, W, B format  
//   btn: the pre-calculated popcnt(W2) of BTN  

 - TAB_Conv2d(torch::Tensor X, torch::Tensor QW, torch::Tensor thresholds, torch::Tensor btn, int type, int padding1, int padding2, int stride1, int stride2, int N, int H, int W, int C, int KN, int KH, int KW)
   - type code: 0: TNN, 1: TBN, 2: BTN, 3: BNN
   - btn is the pre-calculated  popcnt(W2) of BTN
   - It returns the output feature maps just as normal Conv2d functions in PyTorch

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

