from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='tab_cpp',
    version='1.0',
    description='The working version of TAB',
    ext_modules=[
        CppExtension(name='tab_cpp',
                     sources=['TAB_CPU.cpp'],
                     extra_compile_args=['/openmp /arch:AVX2 /Ot']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
