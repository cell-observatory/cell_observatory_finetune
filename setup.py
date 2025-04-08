import os
import glob

from setuptools import setup, find_packages

import numpy
import torch

from torch.utils import cpp_extension
from torch.utils.cpp_extension import CUDAExtension, CppExtension, CUDA_HOME

if 'CONDA_PREFIX' in os.environ:
    conda_bin = os.path.join(os.environ['CONDA_PREFIX'], 'bin')
    os.environ['CC'] = 'x86_64-conda-linux-gnu-gcc'
    os.environ['CXX'] = 'x86_64-conda-linux-gnu-g++'

def print_compile_env():
    import subprocess
    print('torch :', torch.__version__)
    print('torch.cuda :', torch.version.cuda)
    print("CUDA_HOME :", CUDA_HOME)
    try:
        with open(os.devnull, 'w') as devnull:
            gcc = subprocess.check_output(['gcc', '--version'],
                                          stderr=devnull).decode().rstrip('\r\n').split('\n')[0]
        print('gcc :', gcc)
    except Exception as e:
        pass


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, 'csrc')

    # Find all .cpp and .cu files in csrc directory:
    main_files = glob.glob(os.path.join(extensions_dir, '*.cpp'))
    source_cuda = glob.glob(os.path.join(extensions_dir, 'cuda', '*.cu'))

    # sources = main_files
    # source_cuda = glob.glob(os.path.join(extensions_dir, 'cuda', '*.cu'))

    sources = [os.path.relpath(f, start=this_dir) for f in main_files]
    source_cuda = [os.path.relpath(f, start=this_dir) for f in source_cuda]

    sources = sources + source_cuda

    extension = CppExtension

    define_macros = []
    extra_compile_args = {}

    if torch.cuda.is_available() and CUDA_HOME is not None:
        extension = CUDAExtension
        define_macros += [('WITH_CUDA', None)]
        nvcc_flags = os.getenv('NVCC_FLAGS', '').split() if os.getenv('NVCC_FLAGS') else []

        extra_compile_args = {
            'cxx': ['-O2'],  # or '-O0' for debugging
            'nvcc': nvcc_flags + ['-allow-unsupported-compiler'],
        }

    # sources = [os.path.join(extensions_dir, s) for s in sources]
    # include_dirs = [extensions_dir, numpy.get_include()]

    include_dirs = [os.path.relpath(extensions_dir, start=this_dir), numpy.get_include()]

    ext_modules = [
        extension(
            'ops3d._C',  
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            libraries=['curand'],  
        )
    ]

    return ext_modules


if __name__ == "__main__":

    os.makedirs("src/ops3d", exist_ok=True)
    
    # Create an empty __init__.py file if it doesn't exist.
    init_path = os.path.join("src/ops3d", "__init__.py")
    if not os.path.exists(init_path):
        with open(init_path, "w") as f:
            f.write("")

    print_compile_env()

    setup(
        name="segmentation",
        version='0.1.0',
        description='Custom 3D Operations for PyTorch',
        # packages=find_packages(),
        ext_modules=get_extensions(),
        cmdclass={'build_ext': cpp_extension.BuildExtension},
        classifiers=[
            "Programming Language :: Python :: 3",
        ],
        zip_safe=True,
    )
