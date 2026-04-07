"""SparseOcc_ori setup.py — CUDA 확장 포함 빌드.

원본의 models/csrc/setup.py를 프로젝트 루트로 이동.
설치: pip install -e . 또는 python setup.py build_ext --inplace
"""
import os
import torch
from setuptools import find_packages, setup
from os import path as osp
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension


def make_cuda_ext(name, module, sources, sources_cuda=None, extra_args=None, extra_include_path=None):
    if sources_cuda is None:
        sources_cuda = []
    if extra_args is None:
        extra_args = []
    if extra_include_path is None:
        extra_include_path = []

    define_macros = []
    extra_compile_args = {'cxx': list(extra_args)}

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtension
        extra_compile_args['nvcc'] = list(extra_args) + [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
        ]
        sources = list(sources) + list(sources_cuda)
    else:
        print('Compiling {} without CUDA'.format(name))
        extension = CppExtension

    module_dir = osp.join(*module.split('.'))
    return extension(
        name='{}.{}'.format(module, name),
        sources=[osp.join(module_dir, p) for p in sources],
        include_dirs=extra_include_path,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
    )


# CUDA msmv_sampling 확장 (원본 models/csrc/setup.py 기반)
ext_modules = []
msmv_src = osp.join('sparseocc_eccv', 'models', 'csrc', 'msmv_sampling')
if osp.exists(msmv_src):
    try:
        ext_modules.append(make_cuda_ext(
            name='_msmv_sampling_cuda',
            module='sparseocc_eccv.models.csrc',
            sources=['msmv_sampling/msmv_sampling.cpp'],
            sources_cuda=[
                'msmv_sampling/msmv_sampling_forward.cu',
                'msmv_sampling/msmv_sampling_backward.cu',
            ],
            extra_include_path=[osp.abspath(msmv_src)],
        ))
    except Exception as e:
        print(f'Warning: CUDA extension build skipped: {e}')

setup(
    name='SparseOcc_eccv',
    version='0.1.0',
    description='SparseOcc_eccv — migrated to mmdet3d 1.x / mmengine',
    packages=find_packages(exclude=['configs', 'tools', 'tests*']),
    include_package_data=True,
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension} if ext_modules else {},
    python_requires='>=3.7',
    install_requires=[
        'torch',
        'mmengine',
        'mmdet3d',
    ],
)
