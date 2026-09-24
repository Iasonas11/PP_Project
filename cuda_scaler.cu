#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <cuda_runtime.h>

// Macro για έλεγχο λαθών κατά την κλήση συναρτήσεων CUDA
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// ==========================================================
// Custom Atomics για double
// ==========================================================
__device__ void atomicMinDouble(double* address, double val) {
    unsigned long long* address_as_ull = (unsigned long long*)address;
    unsigned long long old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(fmin(val, __longlong_as_double(assumed))));
    } while (assumed != old);
}

__device__ void atomicMaxDouble(double* address, double val) {
    unsigned long long* address_as_ull = (unsigned long long*)address;
    unsigned long long old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(fmax(val, __longlong_as_double(assumed))));
    } while (assumed != old);
}

// ==========================================================
// KERNELS
// ==========================================================

// Αρχικοποίηση των Min/Max πινάκων στη μνήμη της GPU
__global__ void init_minmax_kernel(double* min_v, double* max_v, int D) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < D) {
        min_v[col] = DBL_MAX;
        max_v[col] = -DBL_MAX;
    }
}

// Kernel Φάσης 1: Υπολογισμός μερικών αθροισμάτων και Min/Max
__global__ void compute_stats_kernel(const double* d_buffer, double* d_sum, double* d_sum_sq, double* d_min, double* d_max, int block_rows, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = block_rows * D;

    if (idx < total_elements) {
        int col = idx % D;
        double val = d_buffer[idx];

        // H Tesla V100 υποστηρίζει εγγενώς atomicAdd για double
        atomicAdd(&d_sum[col], val);
        atomicAdd(&d_sum_sq[col], val * val);
        
        atomicMinDouble(&d_min[col], val);
        atomicMaxDouble(&d_max[col], val);
    }
}

// Kernel Φάσης 2: Εφαρμογή του Μετασχηματισμού
__global__ void scale_kernel(double* d_buffer, const double* d_shift, const double* d_scale, int block_rows, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = block_rows * D;

    if (idx < total_elements) {
        int col = idx % D;
        d_buffer[idx] = (d_buffer[idx] - d_shift[col]) * d_scale[col];
    }
}

// ==========================================================
// MAIN FUNCTION
// ==========================================================
int main(int argc, char* argv[]) {
    if (argc < 6 || argc > 7) {
        fprintf(stderr, "Usage: ./cuda_scaler <input.bin> <output.bin> <N> <D> <mode> [block_rows]\n");
        return 1;
    }

    const char* input_file = argv[1];
    const char* output_file = argv[2];
    size_t N = strtoull(argv[3], NULL, 10);
    size_t D = strtoull(argv[4], NULL, 10);
    const char* mode = argv[5];
    size_t block_rows = (argc == 7) ? strtoull(argv[6], NULL, 10) : 100000;

    // CPU Pointers
    double *h_sum, *h_sum_sq, *h_min, *h_max, *h_buffer;
    h_sum = (double*)calloc(D, sizeof(double));
    h_sum_sq = (double*)calloc(D, sizeof(double));
    h_min = (double*)malloc(D * sizeof(double));
    h_max = (double*)malloc(D * sizeof(double));
    
    h_buffer = (double*)malloc(block_rows * D * sizeof(double));

    // GPU Pointers
    double *d_sum, *d_sum_sq, *d_min, *d_max, *d_buffer;
    CUDA_CHECK(cudaMalloc(&d_sum, D * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sum_sq, D * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_min, D * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_max, D * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_buffer, block_rows * D * sizeof(double)));

    // Αρχικοποίηση στοιχείων στην GPU
    CUDA_CHECK(cudaMemset(d_sum, 0, D * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_sum_sq, 0, D * sizeof(double)));
    
    int threadsPerBlock = 256;
    int blocks_D = (D + threadsPerBlock - 1) / threadsPerBlock;
    init_minmax_kernel<<<blocks_D, threadsPerBlock>>>(d_min, d_max, D);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ==========================================================
    // ΦΑΣΗ 1: Υπολογισμός Στατιστικών (CUDA)
    // ==========================================================
    FILE* fin = fopen(input_file, "rb");
    if (!fin) return 1;

    size_t rows_processed = 0;
    while (rows_processed < N) {
        int current_block_rows = (N - rows_processed < block_rows) ? (N - rows_processed) : block_rows;
        size_t elements_to_rw = current_block_rows * D;

        fread(h_buffer, sizeof(double), elements_to_rw, fin);

        // Μεταφορά: CPU -> GPU
        CUDA_CHECK(cudaMemcpy(d_buffer, h_buffer, elements_to_rw * sizeof(double), cudaMemcpyHostToDevice));

        // Εκτέλεση Kernel
        int blocks = (elements_to_rw + threadsPerBlock - 1) / threadsPerBlock;
        compute_stats_kernel<<<blocks, threadsPerBlock>>>(d_buffer, d_sum, d_sum_sq, d_min, d_max, current_block_rows, D);
        CUDA_CHECK(cudaDeviceSynchronize());

        rows_processed += current_block_rows;
    }

    // Φέρνουμε τα τελικά στατιστικά πίσω στη CPU
    CUDA_CHECK(cudaMemcpy(h_sum, d_sum, D * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_sum_sq, d_sum_sq, D * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_min, d_min, D * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_max, d_max, D * sizeof(double), cudaMemcpyDeviceToHost));

    // Η CPU υπολογίζει τα shift_v και scale_v
    double* h_shift = (double*)malloc(D * sizeof(double));
    double* h_scale = (double*)malloc(D * sizeof(double));

    for (size_t j = 0; j < D; ++j) {
        double mean = h_sum[j] / N;
        double variance = (h_sum_sq[j] / N) - (mean * mean);
        if (variance < 0.0) variance = 0.0;
        double std = sqrt(variance);

        if (strcmp(mode, "standard") == 0) {
            h_shift[j] = mean;
            h_scale[j] = (std == 0.0) ? 0.0 : (1.0 / std);
        } else {
            h_shift[j] = h_min[j];
            h_scale[j] = (h_max[j] == h_min[j]) ? 0.0 : (1.0 / (h_max[j] - h_min[j]));
        }
    }

    // ==========================================================
    // ΦΑΣΗ 2: Εφαρμογή Μετασχηματισμών (CUDA)
    // ==========================================================
    
    // Στέλνουμε τα shift και scale στην GPU
    double *d_shift, *d_scale;
    CUDA_CHECK(cudaMalloc(&d_shift, D * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_scale, D * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_shift, h_shift, D * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scale, h_scale, D * sizeof(double), cudaMemcpyHostToDevice));

    fseek(fin, 0, SEEK_SET); 
    FILE* fout = fopen(output_file, "wb");

    rows_processed = 0;
    while (rows_processed < N) {
        int current_block_rows = (N - rows_processed < block_rows) ? (N - rows_processed) : block_rows;
        size_t elements_to_rw = current_block_rows * D;

        // Διαβάζουμε από τον δίσκο (CPU)
        fread(h_buffer, sizeof(double), elements_to_rw, fin);

        // Μεταφορά προς GPU
        CUDA_CHECK(cudaMemcpy(d_buffer, h_buffer, elements_to_rw * sizeof(double), cudaMemcpyHostToDevice));

        // Εκτέλεση Scaling στην GPU
        int blocks = (elements_to_rw + threadsPerBlock - 1) / threadsPerBlock;
        scale_kernel<<<blocks, threadsPerBlock>>>(d_buffer, d_shift, d_scale, current_block_rows, D);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Μεταφορά τροποποιημένου block πίσω στη CPU
        CUDA_CHECK(cudaMemcpy(h_buffer, d_buffer, elements_to_rw * sizeof(double), cudaMemcpyDeviceToHost));

        // Εγγραφή στον δίσκο (CPU)
        fwrite(h_buffer, sizeof(double), elements_to_rw, fout);

        rows_processed += current_block_rows;
    }

    // Καθαρισμός
    fclose(fin);
    fclose(fout);
    free(h_sum); free(h_sum_sq); free(h_min); free(h_max); free(h_shift); free(h_scale);
    free(h_buffer); 
    CUDA_CHECK(cudaFree(d_sum)); CUDA_CHECK(cudaFree(d_sum_sq)); 
    CUDA_CHECK(cudaFree(d_min)); CUDA_CHECK(cudaFree(d_max)); 
    CUDA_CHECK(cudaFree(d_buffer)); CUDA_CHECK(cudaFree(d_shift)); CUDA_CHECK(cudaFree(d_scale));

    return 0;
}