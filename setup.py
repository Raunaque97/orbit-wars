import sys

from setuptools import Extension, setup


class Pybind11Include:
    def __str__(self):
        import pybind11

        return pybind11.get_include()


def _torch_extension_kwargs():
    try:
        import torch
        from torch.utils.cpp_extension import include_paths, library_paths
    except Exception:
        return {}

    extra_link_args = [
        "-ltorch",
        "-ltorch_cpu",
        "-ltorch_python",
        "-lc10",
        "-Wl,-rpath," + library_paths()[0],
        "-Wl,-rpath,@loader_path/../.venv/lib/python"
        + f"{sys.version_info.major}.{sys.version_info.minor}"
        + "/site-packages/torch/lib",
    ]
    if torch.backends.mps.is_built():
        extra_link_args.extend(["-framework", "Accelerate"])
    return {
        "include_dirs": include_paths(),
        "library_dirs": library_paths(),
        "extra_link_args": extra_link_args,
    }


torch_kwargs = _torch_extension_kwargs()


ext_modules = [
    Extension(
        "orbit_native",
        [
            "src/orbit_native/bindings.cpp",
            "src/orbit_native/orbit_core.cpp",
        ],
        include_dirs=[Pybind11Include(), "src/orbit_native"],
        language="c++",
        extra_compile_args=["-std=c++17", "-O3"],
    ),
    Extension(
        "orbit_rl_native",
        [
            "rl/bindings.cpp",
            "rl/rl_features.cpp",
        ],
        include_dirs=[Pybind11Include(), "rl"],
        language="c++",
        extra_compile_args=["-std=c++17", "-O3"],
    ),
    Extension(
        "orbit_rollout_native",
        [
            "rl/rollout_bindings.cpp",
            "rl/rl_features.cpp",
            "src/orbit_native/orbit_core.cpp",
        ],
        include_dirs=[
            Pybind11Include(),
            "rl",
            "src/orbit_native",
            *torch_kwargs.get("include_dirs", []),
        ],
        library_dirs=torch_kwargs.get("library_dirs", []),
        language="c++",
        extra_compile_args=["-std=c++17", "-O3"],
        extra_link_args=torch_kwargs.get("extra_link_args", []),
    ),
]


setup(ext_modules=ext_modules)
