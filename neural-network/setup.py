#!/usr/bin/env python3
"""
Setup script for CUDA-accelerated neural network
"""

import torch
from torch.utils.cpp_extension import CUDAExtension, build_ext
from setuptools import setup, Extension
import os

# Check if CUDA is available
if not torch.cuda.is_available():
    print("CUDA not available, building CPU-only version")
    cuda_extensions = []
else:
    print(f"CUDA available, building with CUDA support")
    cuda_extensions = [
        CUDAExtension(
            name='cuda_kernels',
            sources=['cuda_kernels.cu'],
            include_dirs=[
                os.path.join(torch.utils.cpp_extension.CUDA_HOME, 'include'),
                os.path.join(torch.utils.cpp_extension.CUDA_HOME, 'include', 'cublas_v2.h'),
            ],
            libraries=['cublas', 'curand'],
            extra_compile_args={
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '--ptxas-options=-v',
                    '--gpu-architecture=compute_70',
                    '--gpu-code=compute_70,sm_70,sm_75,sm_80,sm_86,sm_90',
                    '-Xptxas', '-O3',
                    '-Xcompiler', '-O3',
                    '-lineinfo',
                    '--expt-relaxed-constexpr',
                    '--expt-extended-lambda',
                ],
                'cxx': ['-O3', '-fopenmp']
            },
            extra_link_args=['-lcublas', '-lcurand']
        )
    ]

setup(
    name='neural_network_cuda',
    version='1.0.0',
    description='CUDA-accelerated neural network with transformer architecture',
    author='Neural Network Team',
    ext_modules=cuda_extensions,
    cmdclass={'build_ext': build_ext},
    python_requires='>=3.8',
    install_requires=[
        'torch>=1.12.0',
        'numpy>=1.21.0',
        'transformers>=4.20.0',
        'tokenizers>=0.12.0',
        'sentencepiece>=0.1.96',
        'tqdm>=4.64.0',
        'wandb>=0.12.0',
        'tensorboard>=2.8.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0.0',
            'black>=22.0.0',
            'flake8>=4.0.0',
            'mypy>=0.950',
        ],
        'training': [
            'datasets>=2.0.0',
            'accelerate>=0.12.0',
            'deepspeed>=0.6.0',
        ]
    }
)
