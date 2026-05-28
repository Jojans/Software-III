# ILP-Driven Supervised Learning for Co-Optimizing Performance and Energy in CPU-GPU Architectures

> Companion repository for the paper submitted to **CARLA 2026** — Latin American High Performance Computing Conference.  
> Universidad Industrial de Santander (UIS) — SC3UIS Research Group

---

## Overview

This repository contains the full experimental infrastructure for a hybrid **ILP+ML** scheduling methodology for heterogeneous CPU-GPU systems. The proposed approach uses Integer Linear Programming (ILP) to generate provably optimal CPU-GPU task assignments, which are then used as labeled training data for supervised classification models (Random Forest and XGBoost). At runtime, the trained models predict near-optimal assignments in near-constant time, enabling energy-aware scheduling without the computational overhead of solving an ILP at every decision point.

The methodology is validated on three representative HPC computational kernels:

- **GEMM** — General Matrix-Matrix Multiplication (compute-bound)
- **SpMV** — Sparse Matrix-Vector Multiplication (memory-bound)
- **FFT** — Fast Fourier Transform (mixed intensity)

Experiments were conducted on two hardware platforms at the SC3 HPC facility of UIS:

| Platform | CPU | GPU |
|---|---|---|
| HPE ProLiant XL290n G10+ | Intel Xeon Gold 5315Y | NVIDIA A100 40GB |
| HPE ProLiant DL580 G9 | Intel Xeon E5-4640 | NVIDIA Tesla M40 24GB |

---

## Repository Structure

```
.
├── benchmark_gpu_nvidia.cu     # GPU kernel benchmarks (GEMM, SpMV, FFT) — CUDA/cuBLAS/cuSPARSE/cuFFT
├── benchmark_cpu_intel.cpp     # CPU kernel benchmarks for Intel nodes — MKL/FFTW3
├── benchmark_cpu_amd.cpp       # CPU kernel benchmarks for AMD nodes — OpenBLAS/Eigen/FFTW3
├── benchmark_amd.hip.cpp       # GPU kernel benchmarks for AMD — HIP/rocBLAS/rocSPARSE/rocFFT
├── ILP_pacca.py                # ILP solver for HPE XL290n G10+ (NVIDIA A100) platform
├── ILP_thor.py                 # ILP solver for HPE DL580 G9 (Tesla M40) platform
├── ML_models.py                # ML training pipeline — Random Forest and XGBoost with Optuna HPO
├── results_pacca_labeled.csv   # Labeled dataset for PACCA platform (ILP-annotated)
├── results_thor_labeled.csv    # Labeled dataset for THOR platform (ILP-annotated)
└── README.md
```

---

## Benchmarks

Each benchmark measures **execution time** and **energy consumption** for GEMM, SpMV, and FFT across a range of problem sizes, CPU core configurations, and (for SpMV) matrix sparsity levels.

- **`benchmark_gpu_nvidia.cu`** — Implements GEMM via `cublasDgemm`, SpMV via `cusparseSpMV`, and FFT via `cuFFT`. Energy is sampled via NVML at 10 ms intervals from a dedicated pthread.
- **`benchmark_cpu_intel.cpp`** — Uses Intel MKL (`cblas_dgemm`, Inspector-Executor SpMV, DFTI) with AVX-512 vectorization. Energy via Intel RAPL sysfs.
- **`benchmark_cpu_amd.cpp`** — Uses OpenBLAS for GEMM, Eigen for SpMV, and FFTW3 for FFT on AMD EPYC nodes. Energy via AMD sysfs hwmon.
- **`benchmark_amd.hip.cpp`** — HIP/ROCm implementation using rocBLAS, rocSPARSE, and rocFFT for AMD Instinct MI210. Energy via AMD-SMI or sysfs hwmon.

All benchmarks follow the same measurement protocol: **3 warm-up runs** (discarded) followed by **5 measured repetitions**, reporting the median execution time and integrated energy.

---

## ILP Solver

The ILP model formulates CPU-GPU task assignment as a binary optimization problem minimizing a weighted combination of total execution latency and energy consumption:

$$\min\ Z(\mathbf{x}) = Z_L(\mathbf{x}) + \alpha \cdot \sigma \cdot Z_E(\mathbf{x})$$

with $\alpha = 0.3$ (energy policy weight) and hardware constraints on GPU memory capacity, memory bandwidth, and maximum GPU occupancy. The model is solved offline using **PuLP with the CBC solver**.

- **`ILP_pacca.py`** — Configured for the HPE XL290n G10+ node (NVIDIA A100, 40 GB VRAM, 1555 GB/s bandwidth).
- **`ILP_thor.py`** — Configured for the HPE DL580 G9 node (Tesla M40, 24 GB VRAM, 288 GB/s bandwidth).

Both scripts read a raw results CSV, solve the ILP for each CPU-core configuration, and output a labeled dataset with the optimal binary assignment (`ilp_label`: 0 = CPU, 1 = GPU) appended as a column.

**Usage:**
```bash
python ILP_pacca.py --input results_pacca.csv
python ILP_thor.py  --input results_thor.csv
```

---

## ML Pipeline

**`ML_models.py`** trains and evaluates two ensemble classifiers on the ILP-labeled dataset:

- **Random Forest** — Bootstrap aggregation with random feature subsets (MDI importance).
- **XGBoost** — Additive gradient boosting with second-order Taylor approximation of cross-entropy loss.

Hyperparameter optimization is performed via **Optuna TPE** (100 trials per model, 5-fold stratified CV, ROC-AUC as objective). The script produces the following outputs:

| Output | Description |
|---|---|
| `fig1_model_evaluation.png` | Confusion matrices, ROC curves, PR curves, metric comparison |
| `fig2_feature_importance.png` | RF MDI importance vs. XGBoost importance by gain |
| `fig3_learning_dynamics.png` | Learning curves and per-kernel accuracy |
| `fig4_confidence_analysis.png` | Prediction confidence vs. speedup ratio |
| `fig5_optuna_history.png` | Optuna optimization history and fANOVA hyperparameter importance |

**Usage:**
```bash
python ML_models.py --input results.csv
```

---

## Datasets

The labeled CSV files contain one row per experimental observation with the following columns:

| Column | Description |
|---|---|
| `kernel` | Kernel type: GEMM, SpMV, FFT |
| `N` | Problem size |
| `sparsity` | Sparsity fraction (SpMV only) |
| `cpu_cores` | Number of active CPU cores |
| `cpu_time_s` | Median CPU execution time (seconds) |
| `gpu_time_s` | Median GPU execution time (seconds) |
| `cpu_energy_j` | CPU energy consumption (Joules) |
| `gpu_energy_j` | GPU energy consumption (Joules) |
| `speedup` | Ratio $T_\text{cpu} / T_\text{gpu}$ |
| `op_intensity_cpu` | Operational intensity on CPU (FLOP/byte) |
| `op_intensity_gpu` | Operational intensity on GPU (FLOP/byte) |
| `ilp_label` | ILP-optimal assignment: 0 = CPU, 1 = GPU |

---

## Requirements

**Python:**
```
pulp
xgboost
scikit-learn
optuna
pandas
numpy
matplotlib
seaborn
```

**C++/CUDA (per platform):**
- CUDA 12.x + cuBLAS + cuSPARSE + cuFFT + NVML
- ROCm 7.2 + rocBLAS + rocSPARSE + rocFFT
- Intel oneAPI MKL
- OpenBLAS + Eigen + FFTW3 (AMD CPU nodes)
- GCC 14.2 / icpx / nvcc / hipcc

---

## Citation

If you use this code or dataset in your work, please cite:

```bibtex
@inproceedings{lemus2026ilpml,
  title     = {ILP-Driven Supervised Learning for Co-Optimizing Performance
               and Energy in CPU-GPU Architectures},
  author    = {Lemus Ram{\'i}rez, Anderson Jahir and
               Galvis Beltr{\'a}n, Johan Sebastian and
               Torres Ni{\~n}o, Luis Alejandro and
               Jaimes Barrios Hernandez, Carlos},
  booktitle = {Proceedings of CARLA 2026 -- Latin American High Performance
               Computing Conference},
  series    = {Communications in Computer and Information Science},
  publisher = {Springer},
  year      = {2026}
}
```

---

## Authors

- **Anderson Jahir Lemus Ramírez** — Universidad Industrial de Santander / SC3UIS
- **Johan Sebastian Galvis Beltrán** — Universidad Industrial de Santander / SC3UIS
- **Luis Alejandro Torres Niño** — Universidad Industrial de Santander / SC3UIS
- **Carlos Jaimes Barrios Hernandez** — Universidad Industrial de Santander / SC3UIS / LIG-INRIA Grenoble / INSA Lyon

---

*This work was carried out using the HPC infrastructure of the SC3UIS research group at the Universidad Industrial de Santander, Bucaramanga, Colombia.*
