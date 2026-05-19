from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="selector_paged_pq",
    version="0.1",
    packages=["selector_paged_pq"],
    ext_modules=[
        CUDAExtension(
            "selector_paged_pq._C",
            sources=[
                "paged_pq_ext.cpp",
                "paged_pq_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "-std=c++17", "--expt-relaxed-constexpr"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.10",
)
