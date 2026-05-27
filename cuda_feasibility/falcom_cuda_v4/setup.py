from pathlib import Path

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


TORCH_LIB = Path(torch.__file__).resolve().parent / "lib"

setup(
    name="falcom_cuda_v4",
    ext_modules=[
        CUDAExtension(
            name="_falcom_cuda_v4",
            sources=["falcom_cuda_v4.cpp", "falcom_cuda_v4_kernels.cu"],
            extra_link_args=[f"-Wl,-rpath,{TORCH_LIB}"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-std=c++17",
                    "-gencode=arch=compute_70,code=sm_70",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
