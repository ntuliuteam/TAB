from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


setup(
    name='tab_cpp',
    version='1.0',
    description='The working version of TAB',
    ext_modules=[
        CppExtension(name='tab_cpp',
                     sources=['TAB_CPU.cpp'],
                     extra_compile_args=['-fopenmp -flax-vector-conversions -march=armv8-a+simd -mfpu=neon -funsafe-math-optimizations -fbuiltin -O3']),
        
    ],
    cmdclass={
        'build_ext': BuildExtension
    })