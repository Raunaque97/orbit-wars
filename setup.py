from setuptools import Extension, setup


class Pybind11Include:
    def __str__(self):
        import pybind11

        return pybind11.get_include()


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
    )
]


setup(ext_modules=ext_modules)
