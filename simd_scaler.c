#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

// Αφαιρέσαμε το <immintrin.h> και βάλαμε τη βιβλιοθήκη μετάφρασης SIMDe
#include <immintrin.h>

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

    if (strcmp(mode, "standard") != 0 && strcmp(mode, "minmax") != 0) {
        fprintf(stderr, "Error: Mode must be 'standard' or 'minmax'.\n");
        return 1;
    }

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

    // ==========================================================
    // ΦΑΣΗ 1: Υπολογισμός Στατιστικών (SIMD AVX2)
    // ==========================================================
    FILE* fin = fopen(input_file, "rb");
    if (!fin) return 1;

    size_t rows_processed = 0;
    while (rows_processed < N) {
        size_t current_block_rows = (N - rows_processed < block_rows) ? (N - rows_processed) : block_rows;
        size_t elements_to_read = current_block_rows * D;

        fread(buffer, sizeof(double), elements_to_read, fin);

        for (size_t i = 0; i < current_block_rows; ++i) {
            size_t j = 0;
            // SIMD Loop: Επεξεργασία 4 στοιχείων (doubles) ταυτόχρονα
            for (; j + 3 < D; j += 4) {
                // Φόρτωση 4 τιμών από τον buffer
                __m256d val = _mm256_loadu_pd(&buffer[i * D + j]);

                // Ενημέρωση Αθροισμάτων (sum_v += val)
                __m256d sv = _mm256_loadu_pd(&sum_v[j]);
                _mm256_storeu_pd(&sum_v[j], _mm256_add_pd(sv, val));

                // Ενημέρωση Αθροισμάτων Τετραγώνων (sum_sq_v += val * val)
                __m256d val_sq = _mm256_mul_pd(val, val);
                __m256d ssqv = _mm256_loadu_pd(&sum_sq_v[j]);
                _mm256_storeu_pd(&sum_sq_v[j], _mm256_add_pd(ssqv, val_sq));

                // Ενημέρωση Ελαχίστου
                __m256d min_val = _mm256_loadu_pd(&min_v[j]);
                _mm256_storeu_pd(&min_v[j], _mm256_min_pd(min_val, val));

                // Ενημέρωση Μεγίστου
                __m256d max_val = _mm256_loadu_pd(&max_v[j]);
                _mm256_storeu_pd(&max_v[j], _mm256_max_pd(max_val, val));
            }
            
            // Remainder Loop: Για τυχόν στήλες που περισσεύουν (αν το D δεν είναι πολλαπλάσιο του 4)
            for (; j < D; ++j) {
                double val = buffer[i * D + j];
                sum_v[j] += val;
                sum_sq_v[j] += val * val;
                if (val < min_v[j]) min_v[j] = val;
                if (val > max_v[j]) max_v[j] = val;
            }
        }
        rows_processed += current_block_rows;
    }

    // Τελικός Υπολογισμός Μέσης Τιμής & Απόκλισης
    for (size_t j = 0; j < D; ++j) {
        mean_v[j] = sum_v[j] / N;
        double variance = (sum_sq_v[j] / N) - (mean_v[j] * mean_v[j]);
        if (variance < 0.0) variance = 0.0; 
        std_v[j] = sqrt(variance);
    }

    // ==========================================================
    // ΦΑΣΗ 2: Εφαρμογή Μετασχηματισμών (SIMD AVX2)
    // ==========================================================
    
    // Προ-υπολογισμός παραμέτρων μετατόπισης (shift) και κλίμακας (scale) 
    // Αυτό μετατρέπει την αργή διαίρεση σε γρήγορο SIMD πολλαπλασιασμό!
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
    if (!fout) return 1;

    rows_processed = 0;
    while (rows_processed < N) {
        size_t current_block_rows = (N - rows_processed < block_rows) ? (N - rows_processed) : block_rows;
        size_t elements_to_read = current_block_rows * D;

        fread(buffer, sizeof(double), elements_to_read, fin);

        for (size_t i = 0; i < current_block_rows; ++i) {
            size_t j = 0;
            // SIMD Loop
            for (; j + 3 < D; j += 4) {
                __m256d val = _mm256_loadu_pd(&buffer[i * D + j]);
                __m256d shift = _mm256_loadu_pd(&shift_v[j]);
                __m256d scale = _mm256_loadu_pd(&scale_v[j]);

                // Πράξη: val = (val - shift) * scale
                val = _mm256_sub_pd(val, shift);
                val = _mm256_mul_pd(val, scale);

                _mm256_storeu_pd(&buffer[i * D + j], val);
            }
            // Remainder Loop
            for (; j < D; ++j) {
                buffer[i * D + j] = (buffer[i * D + j] - shift_v[j]) * scale_v[j];
            }
        }

        fwrite(buffer, sizeof(double), elements_to_read, fout);
        rows_processed += current_block_rows;
    }

    fclose(fin);
    fclose(fout);
    free(sum_v); free(sum_sq_v); free(min_v); free(max_v); 
    free(mean_v); free(std_v); free(buffer); free(shift_v); free(scale_v);

    return 0;
}
