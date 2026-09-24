# Ορισμός Compilers
CC = gcc
MPICC = mpicc
NVCC = /usr/local/cuda-12.2/bin/nvcc  

# Ορισμός Flags (Βελτιστοποίηση -O3, Εμφάνιση Warnings -Wall)
CFLAGS = -O3 -Wall
LDFLAGS = -lm

# Το 'all' είναι ο προεπιλεγμένος στόχος.
all: serial simd omp mpi cuda

# ==========================================
# Στόχοι Μεταγλώττισης (Compilation Rules)
# ==========================================

serial: serial_scaler.c
	$(CC) $(CFLAGS) serial_scaler.c -o serial_scaler $(LDFLAGS)

simd: simd_scaler.c
	$(CC) $(CFLAGS) -mavx2 simd_scaler.c -o simd_scaler $(LDFLAGS)

omp: omp_scaler.c
	$(CC) $(CFLAGS) -fopenmp omp_scaler.c -o omp_scaler $(LDFLAGS)

mpi: mpi_scaler.c
	$(MPICC) $(CFLAGS) mpi_scaler.c -o mpi_scaler $(LDFLAGS)

cuda: cuda_scaler.cu
	$(NVCC) -O3 -arch=sm_70 cuda_scaler.cu -o cuda_scaler

# ==========================================
# Καθαρισμός
# ==========================================
clean:
	rm -f serial_scaler simd_scaler omp_scaler mpi_scaler cuda_scaler