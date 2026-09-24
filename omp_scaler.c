#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <omp.h>

int main(int argc, char* argv[]) {
    if (argc < 6 || argc > 7) {
        fprintf(stderr, "Usage: ./scaler <input.bin> <output.bin> <N> <D> <mode> [block_rows]\n");
        return 1;
    }

    const char* input_file = argv[1];
    const char* output_file = argv[2];
    size_t N = strtoull(argv[3], NULL, 10);
    size_t D = strtoull(argv[4], NULL, 10);
    const char* mode = argv[5];
    size_t block_rows = (argc == 7) ? strtoull(argv[6], NULL, 10) : 100000;

    double* sum_v = (double*)calloc(D, sizeof(double));
    double* sum_sq_v = (double*)calloc(D, sizeof(double));
    double* min_v = (double*)malloc(D * sizeof(double));
    double* max_v = (double*)malloc(D * sizeof(double));
    double* mean_v = (double*)malloc(D * sizeof(double));
    double* std_v = (double*)malloc(D * sizeof(double));

    for (size_t j = 0; j < D; j++) {
        min_v[j] = DBL_MAX;
        max_v[j] = -DBL_MAX;
    }

    double* buffer = (double*)malloc(block_rows * D * sizeof(double));
    if (!buffer) return 1;

    FILE* fin = fopen(input_file, "rb");
    if (!fin) return 1;

    // ==========================================================
    // ΦΑΣΗ 1: Υπολογισμός Στατιστικών (OpenMP 4.5+ Array Reductions)
    // ==========================================================
    size_t rows_processed = 0;
    while (rows_processed < N) {
        size_t current_block_rows = (N - rows_processed < block_rows) ? (N - rows_processed) : block_rows;
        size_t elements_to_read = current_block_rows * D;

        fread(buffer, sizeof(double), elements_to_read, fin);

        // Το OpenMP αναλαμβάνει αυτόματα τη δημιουργία τοπικών πινάκων και το merge στο τέλος!
        #pragma omp parallel for \
            reduction(+:sum_v[:D], sum_sq_v[:D]) \
            reduction(min:min_v[:D]) \
            reduction(max:max_v[:D])
        for (size_t i = 0; i < current_block_rows; ++i) {
            for (size_t j = 0; j < D; ++j) {
                double val = buffer[i * D + j];
                sum_v[j] += val;
                sum_sq_v[j] += val * val;
                if (val < min_v[j]) min_v[j] = val;
                if (val > max_v[j]) max_v[j] = val;
            }
        }
        rows_processed += current_block_rows;
    }

    for (size_t j = 0; j < D; ++j) {
        mean_v[j] = sum_v[j] / N;
        double variance = (sum_sq_v[j] / N) - (mean_v[j] * mean_v[j]);
        if (variance < 0.0) variance = 0.0;
        std_v[j] = sqrt(variance);
    }

    // ==========================================================
    // ΦΑΣΗ 2: Εφαρμογή Μετασχηματισμών (OpenMP)
    // ==========================================================
    double* shift_v = (double*)malloc(D * sizeof(double));
    double* scale_v = (double*)malloc(D * sizeof(double));
    
    for (size_t j = 0; j < D; ++j) {
        if (strcmp(mode, "standard") == 0) {
            shift_v[j] = mean_v[j];
            scale_v[j] = (std_v[j] == 0.0) ? 0.0 : (1.0 / std_v[j]);
        } else {
            shift_v[j] = min_v[j];
            scale_v[j] = (max_v[j] == min_v[j]) ? 0.0 : (1.0 / (max_v[j] - min_v[j]));
        }
    }

    fseek(fin, 0, SEEK_SET); 
    FILE* fout = fopen(output_file, "wb");

    rows_processed = 0;
    while (rows_processed < N) {
        size_t current_block_rows = (N - rows_processed < block_rows) ? (N - rows_processed) : block_rows;
        size_t elements_to_read = current_block_rows * D;

        fread(buffer, sizeof(double), elements_to_read, fin);

        #pragma omp parallel for
        for (size_t i = 0; i < current_block_rows; ++i) {
            for (size_t j = 0; j < D; ++j) {
                buffer[i * D + j] = (buffer[i * D + j] - shift_v[j]) * scale_v[j];
            }
        }

        fwrite(buffer, sizeof(double), elements_to_read, fout);
        rows_processed += current_block_rows;
    }

    fclose(fin); fclose(fout);
    free(sum_v); free(sum_sq_v); free(min_v); free(max_v); 
    free(mean_v); free(std_v); free(buffer); free(shift_v); free(scale_v);

    return 0;
}