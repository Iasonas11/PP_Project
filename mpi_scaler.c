#define _FILE_OFFSET_BITS 64 // Απαραίτητο για αρχεία > 2GB (Large, Very Large)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <mpi.h> 
#include <sys/types.h>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 6 || argc > 7) {
        if (rank == 0) {
            fprintf(stderr, "Usage: mpiexec -n <procs> ./mpi_scaler <input.bin> <output.bin> <N> <D> <mode> [block_rows]\n");
        }
        MPI_Finalize();
        return 1;
    }

    const char* input_file = argv[1];
    const char* output_file = argv[2];
    size_t N = strtoull(argv[3], NULL, 10);
    size_t D = strtoull(argv[4], NULL, 10);
    const char* mode = argv[5];
    size_t block_rows = (argc == 7) ? strtoull(argv[6], NULL, 10) : 100000;

    size_t local_N = N / size;
    size_t remainder = N % size;
    size_t start_row = rank * local_N + (rank < remainder ? rank : remainder);
    if (rank < remainder) local_N++;

    double* local_sum = (double*)calloc(D, sizeof(double));
    double* local_sum_sq = (double*)calloc(D, sizeof(double));
    double* local_min = (double*)malloc(D * sizeof(double));
    double* local_max = (double*)malloc(D * sizeof(double));
    
    double* sum_v = (double*)calloc(D, sizeof(double));
    double* sum_sq_v = (double*)calloc(D, sizeof(double));
    double* min_v = (double*)malloc(D * sizeof(double));
    double* max_v = (double*)malloc(D * sizeof(double));
    
    for (size_t j = 0; j < D; j++) {
        local_min[j] = DBL_MAX; local_max[j] = -DBL_MAX;
        min_v[j] = DBL_MAX; max_v[j] = -DBL_MAX;
    }

    double* local_buffer = (double*)malloc(block_rows * D * sizeof(double));

    // ==========================================================
    // ΦΑΣΗ 1: Υπολογισμός Στατιστικών (Κλασικό Parallel File I/O)
    // ==========================================================
    FILE* fin = fopen(input_file, "rb");
    if (!fin) {
        if (rank == 0) fprintf(stderr, "Error opening input file.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    size_t rows_processed = 0;
    while (rows_processed < local_N) {
        size_t current_block_rows = (local_N - rows_processed < block_rows) ? (local_N - rows_processed) : block_rows;
        size_t elements_to_read = current_block_rows * D;
        
        // Χρήση fseeko για μετακίνηση στη σωστή θέση (ασφαλές για μεγάλα αρχεία)
        off_t byte_offset = (off_t)(start_row + rows_processed) * D * sizeof(double);
        fseeko(fin, byte_offset, SEEK_SET);
        fread(local_buffer, sizeof(double), elements_to_read, fin);

        for (size_t i = 0; i < current_block_rows; i++) {
            for (size_t j = 0; j < D; j++) {
                double val = local_buffer[i * D + j];
                local_sum[j] += val;
                local_sum_sq[j] += val * val;
                if (val < local_min[j]) local_min[j] = val;
                if (val > local_max[j]) local_max[j] = val;
            }
        }
        rows_processed += current_block_rows;
    }
    fclose(fin);

    MPI_Allreduce(local_sum, sum_v, D, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(local_sum_sq, sum_sq_v, D, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(local_min, min_v, D, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(local_max, max_v, D, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    double* shift_v = (double*)malloc(D * sizeof(double));
    double* scale_v = (double*)malloc(D * sizeof(double));

    for (size_t j = 0; j < D; ++j) {
        double mean = sum_v[j] / N;
        double variance = (sum_sq_v[j] / N) - (mean * mean);
        if (variance < 0.0) variance = 0.0;
        double std = sqrt(variance);

        if (strcmp(mode, "standard") == 0) {
            shift_v[j] = mean;
            scale_v[j] = (std == 0.0) ? 0.0 : (1.0 / std);
        } else {
            shift_v[j] = min_v[j];
            scale_v[j] = (max_v[j] == min_v[j]) ? 0.0 : (1.0 / (max_v[j] - min_v[j]));
        }
    }

    // ==========================================================
    // ΦΑΣΗ 2: Εφαρμογή Μετασχηματισμών (Κλασικό Parallel File I/O)
    // ==========================================================
    
    // Ο Master δημιουργεί το άδειο αρχείο εξόδου ώστε να υπάρχει για όλους
    if (rank == 0) {
        FILE* create_fout = fopen(output_file, "wb");
        if (create_fout) fclose(create_fout);
    }
    // Περιμένουμε όλοι να δημιουργηθεί το αρχείο
    MPI_Barrier(MPI_COMM_WORLD);

    fin = fopen(input_file, "rb");
    // Ανοίγουμε με "r+b" για να κάνουμε update συγκεκριμένα bytes χωρίς να σβήσουμε τα υπόλοιπα
    FILE* fout = fopen(output_file, "r+b"); 

    rows_processed = 0;
    while (rows_processed < local_N) {
        size_t current_block_rows = (local_N - rows_processed < block_rows) ? (local_N - rows_processed) : block_rows;
        size_t elements_to_rw = current_block_rows * D;
        
        off_t byte_offset = (off_t)(start_row + rows_processed) * D * sizeof(double);
        
        fseeko(fin, byte_offset, SEEK_SET);
        fread(local_buffer, sizeof(double), elements_to_rw, fin);

        for (size_t i = 0; i < current_block_rows; i++) {
            for (size_t j = 0; j < D; j++) {
                local_buffer[i * D + j] = (local_buffer[i * D + j] - shift_v[j]) * scale_v[j];
            }
        }

        fseeko(fout, byte_offset, SEEK_SET);
        fwrite(local_buffer, sizeof(double), elements_to_rw, fout);

        rows_processed += current_block_rows;
    }

    fclose(fin);
    if (fout) fclose(fout);

    free(local_sum); free(local_sum_sq); free(local_min); free(local_max);
    free(sum_v); free(sum_sq_v); free(min_v); free(max_v);
    free(local_buffer); free(shift_v); free(scale_v);

    MPI_Finalize();
    return 0;
}