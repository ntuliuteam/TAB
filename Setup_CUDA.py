from setuptools import setup
from torch.utils.cpp_extension import BuildExtension,  CUDAExtension

setup(
    name='TAB_CUDA',
    version='1.0',
    description='The working version of TAB on CUDA',
    ext_modules=[
        CUDAExtension(name='TAB_CUDA',
                     sources=['TAB_GPU.cpp', 'TAB_GEMM.cu'],
                     extra_compile_args=[]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
