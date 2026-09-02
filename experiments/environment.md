# Environment record

Fill this in before collecting any performance data.

| Field | Value |
| --- | --- |
| GPU model | |
| Compute capability | |
| GPU memory | |
| NVIDIA driver version | |
| CUDA Toolkit / `nvcc --version` | |
| OS and version | |
| CMake version | |
| Host compiler/version | |
| Power / clock configuration | |
| Date and time | |

## Verification commands (Windows PowerShell)

```powershell
nvidia-smi
nvcc --version
cmake --version
```

## Build commands

```powershell
cmake -S . -B build
cmake --build build --config Release
.\build\Release\vector_add.exe
.\build\Release\naive_gemm.exe
```

On single-config generators, executables may be directly under `build` rather than `build\\Release`.
