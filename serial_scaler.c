#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

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
    
    // Προαιρετική παράμετρος block_rows (Out-of-Core επεξεργασία)
    size_t block_rows = (argc == 7) ? strtoull(argv[6], NULL, 10) : 100000;

    if (strcmp(mode, "standard") != 0 && strcmp(mode, "minmax") != 0) {
        fprintf(stderr, "Error: Mode must be 'standard' or 'minmax'.\n");
        return 1;
    }

    // Δέσμευση μνήμης για τα στατιστικά ανά στήλη
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

    // Buffer για επεξεργασία block-block
    double* buffer = (double*)malloc(block_rows * D * sizeof(double));
    if (!buffer) {
        fprintf(stderr, "Error: Memory allocation for block buffer failed!\n");
        return 1;
    }

    // ==========================================================
    // ΦΑΣΗ 1: Υπολογισμός Στατιστικών
    // ==========================================================
    FILE* fin = fopen(input_file, "rb");
    if (!fin) {
        fprintf(stderr, "Error opening input file: %s\n", input_file);
        return 1;
    }

    size_t rows_processed = 0;
    while (rows_processed < N) {
        size_t current_block_rows = (N - rows_processed < block_rows) ? (N - rows_processed) : block_rows;
        size_t elements_to_read = current_block_rows * D;

        size_t read_elements = fread(buffer, sizeof(double), elements_to_read, fin);
        if (read_elements != elements_to_read) {
             fprintf(stderr, "Warning: File ended prematurely during Phase 1.\n");
             break;
        }

        // Ενημέρωση αθροισμάτων και min/max
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

    // Υπολογισμός τελικής Μέσης Τιμής και Τυπικής Απόκλισης
    for (size_t j = 0; j < D; ++j) {
        mean_v[j] = sum_v[j] / N;
        double variance = (sum_sq_v[j] / N) - (mean_v[j] * mean_v[j]);
        if (variance < 0.0) variance = 0.0; 
        std_v[j] = sqrt(variance);
    }

    // ==========================================================
    // ΦΑΣΗ 2: Εφαρμογή Μετασχηματισμών
    // ==========================================================
    fseek(fin, 0, SEEK_SET); 
    
    FILE* fout = fopen(output_file, "wb");
    if (!fout) {
        fprintf(stderr, "Error opening output file: %s\n", output_file);
        return 1;
    }

    rows_processed = 0;
    while (rows_processed < N) {
        size_t current_block_rows = (N - rows_processed < block_rows) ? (N - rows_processed) : block_rows;
        size_t elements_to_read = current_block_rows * D;

        fread(buffer, sizeof(double), elements_to_read, fin);

        // Μετασχηματισμός in-place
        for (size_t i = 0; i < current_block_rows; ++i) {
            for (size_t j = 0; j < D; ++j) {
                double val = buffer[i * D + j];
                
                if (strcmp(mode, "standard") == 0) {
                    if (std_v[j] == 0.0) buffer[i * D + j] = 0.0;
                    else buffer[i * D + j] = (val - mean_v[j]) / std_v[j];
                } 
                else if (strcmp(mode, "minmax") == 0) {
                    if (max_v[j] == min_v[j]) buffer[i * D + j] = 0.0;
                    else buffer[i * D + j] = (val - min_v[j]) / (max_v[j] - min_v[j]);
                }
            }
        }

        // Εγγραφή στο νέο αρχείο (row-major)
        fwrite(buffer, sizeof(double), elements_to_read, fout);
        rows_processed += current_block_rows;
    }

    fclose(fin);
    fclose(fout);
    free(sum_v);
    free(sum_sq_v);
    free(min_v);
    free(max_v);
    free(mean_v);
    free(std_v);
    free(buffer);

    return 0;
}