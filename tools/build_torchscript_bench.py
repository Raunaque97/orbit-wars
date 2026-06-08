from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import torch
from torch.utils.cpp_extension import include_paths, library_paths


DEFAULT_SOURCE = Path("tools/bench_torchscript_cpp.cpp")
DEFAULT_OUTPUT = Path("build/bench_torchscript_cpp")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cxx", default="clang++")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        args.cxx,
        "-std=c++17",
        "-O3",
        str(args.source),
        "-o",
        str(args.output),
    ]
    for path in include_paths():
        cmd.extend(["-I", path])
    for path in library_paths():
        cmd.extend(["-L", path])
    cmd.extend(
        [
            "-ltorch",
            "-ltorch_cpu",
            "-lc10",
            "-Wl,-rpath," + library_paths()[0],
        ]
    )
    if torch.backends.mps.is_built():
        # The benchmark itself is CPU-only, but the PyTorch wheel may reference
        # Apple frameworks from its libraries.
        cmd.extend(["-framework", "Accelerate"])

    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    print(f"built={args.output}")


if __name__ == "__main__":
    main()
