# High-Performance Out-of-Core Data Scaler

This repository contains a high-performance, out-of-core implementation of data scaling algorithms (Standard and MinMax). It is designed to process massive datasets that exceed available RAM by reading and writing data in chunks. 

The project explores various levels of parallelization, from vector instructions to distributed memory and GPU acceleration, comparing their performance.

## 🚀 Features

* **Out-of-Core Processing**: Efficiently handles huge binary datasets by processing them in predefined block sizes to avoid Out-Of-Memory (OOM) errors.
* **Multiple Architectures**: 
  * **Serial**: Baseline implementation in C.
  * **SIMD**: Vectorized instructions using AVX2.
  * **OpenMP**: Shared-memory parallelization using array reductions.
  * **MPI**: Distributed-memory parallelization using classic parallel file I/O (`fseeko`).
  * **CUDA**: GPU acceleration using custom atomics for double-precision floats.
* **Automated Validation**: Includes a Python testing suite that verifies the C/CUDA outputs against `scikit-learn` to guarantee numerical correctness (tolerance < 1e-9).
* **Benchmarking Suite**: Bash scripts to automatically generate data, compile, run, and time all implementations across varying dataset sizes and thread/process counts.

## 🛠️ Tech Stack

* **Languages**: C, CUDA C++, Python 3
* **Parallel Computing**: OpenMP, MPI, CUDA Toolkit (nvcc)
* **Validation**: NumPy, scikit-learn

## 📦 Getting Started

### Prerequisites
To compile and run all implementations, you need:
* GCC compiler (with OpenMP & AVX2 support)
* MPI implementation (e.g., OpenMPI or MPICH)
* CUDA Toolkit (specifically `nvcc`, configured for `sm_70` architecture)
* Python 3 with `numpy` and `scikit-learn` (for data generation and correctness checking)

### Compilation
A `Makefile` is provided to compile all targets easily.
```bash
make clean
make all

Running the Experiments

The easiest way to test the project is using the automated benchmarking script. This will generate synthetic data (small, medium, large), run all scalers, and verify the outputs:

Bash
./run_experiments.sh

Manual Execution Example

You can manually generate a dataset and run a specific scaler. For example, to use the CUDA scaler on a 10M rows x 128 features dataset:

Generate Data:
Bash
./run_gendata.sh 10000000 128 data_10M_128.bin

Run CUDA Scaler (Standard Scaling):
Bash
./cuda_scaler data_10M_128.bin out_10M_128.bin 10000000 128 standard 100000

Verify Correctness:
Bash
python3 check_correctness.py --input data_10M_128.bin --output out_10M_128.bin --samples 10000000 --features 128 --mode standard

📂 Project Structure
serial_scaler.c, simd_scaler.c, omp_scaler.c, mpi_scaler.c, cuda_scaler.cu: Scaler implementations.   
generate_data.py: Generates Gaussian-distributed binary datasets out-of-core.
check_correctness.py: Chunk-by-chunk validation against scikit-learn.
run_experiments.sh & run_gendata.sh: Automation scripts
