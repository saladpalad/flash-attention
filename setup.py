from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='flash_attention',
    ext_modules=[
        CUDAExtension('flash_attention', ['flash_att.cu'],
                      extra_compile_args={'cxx': ['-std=c++17'],
                                          'nvcc': ['-std=c++17']})
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
    )
