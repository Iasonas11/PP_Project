#!/bin/bash

# ==========================================
# Script Αυτοματοποιημένης Εκτέλεσης Πειραμάτων
# ==========================================

DATASETS=("small" "medium" "large")
N_ARR=(1000000 5000000 10000000)
D_ARR=(32 64 128)

MODES=("standard" "minmax")
BLOCK=100000

echo "-------------------------------------------------"
echo " Μεταγλώττιση (Compilation)..."
echo "-------------------------------------------------"
make clean
make
echo ""

for i in "${!DATASETS[@]}"; do
    NAME=${DATASETS[$i]}
    N=${N_ARR[$i]}
    D=${D_ARR[$i]}
    INPUT="data_${NAME}.bin"

    echo "================================================================"
    echo " ΕΚΤΕΛΕΣΗ ΠΕΙΡΑΜΑΤΩΝ ΓΙΑ DATASET: ${NAME^^} (N=$N, D=$D)"
    echo "================================================================"

    # Αν δεν υπάρχει το αρχείο, φτιάξτο
    if [ ! -f "$INPUT" ]; then
        echo "[INFO] Δημιουργία δεδομένων $INPUT..."
        ./run_gendata.sh $N $D $INPUT
    fi

    # Loop και για τους δύο τύπους κανονικοποίησης
    for MODE in "${MODES[@]}"; do
        OUTPUT="out_${NAME}_${MODE}.bin"
        
        echo "----------------------------------------------------------------"
        echo " ΔΟΚΙΜΗ SCALER: ${MODE^^}"
        echo "----------------------------------------------------------------"

        echo ">>> 1. Σειριακή Έκδοση (Serial)"
        time -p ./serial_scaler $INPUT $OUTPUT $N $D $MODE $BLOCK
        echo ">>> Έλεγχος Ορθότητας (Serial)..."
        python3 check_correctness.py --input $INPUT --output $OUTPUT --samples $N --features $D --mode $MODE

        echo ""
        echo ">>> 2. SIMD (AVX2)"
        time -p ./simd_scaler $INPUT $OUTPUT $N $D $MODE $BLOCK
        echo ">>> Έλεγχος Ορθότητας (SIMD)..."
        python3 check_correctness.py --input $INPUT --output $OUTPUT --samples $N --features $D --mode $MODE

        echo ""
        echo ">>> 3. OpenMP (Κοινή Μνήμη)"
        # Τρέχουμε το OpenMP για όλα τα threads
        for THREADS in 1 2 4 8 16; do
            echo "- Threads: $THREADS -"
            export OMP_NUM_THREADS=$THREADS
            time -p ./omp_scaler $INPUT $OUTPUT $N $D $MODE $BLOCK
        done
        # Ελέγχουμε την ορθότητα για το τελευταίο παραγόμενο αρχείο (αυτό με τα 16 threads)
        echo ">>> Έλεγχος Ορθότητας (OpenMP 16 Threads)..."
        python3 check_correctness.py --input $INPUT --output $OUTPUT --samples $N --features $D --mode $MODE

        echo ""
        echo ">>> 4. MPI (Κατανεμημένη Μνήμη)"
        for PROCS in 1 2 4 8 16; do
            echo "- Διεργασίες: $PROCS -"
            time -p mpiexec --oversubscribe -n $PROCS ./mpi_scaler $INPUT $OUTPUT $N $D $MODE $BLOCK
        done
        # Ελέγχουμε την ορθότητα για το τελευταίο παραγόμενο αρχείο (αυτό με τις 16 διεργασίες)
        echo ">>> Έλεγχος Ορθότητας (MPI 16 Procs)..."
        python3 check_correctness.py --input $INPUT --output $OUTPUT --samples $N --features $D --mode $MODE
        
        echo ""
        echo ">>> 5. CUDA (GPU)"
        time -p ./cuda_scaler $INPUT $OUTPUT $N $D $MODE $BLOCK
        echo ">>> Έλεγχος Ορθότητας (CUDA)..."
        python3 check_correctness.py --input $INPUT --output $OUTPUT --samples $N --features $D --mode $MODE

        echo ""
    done
done

echo "================================================="
echo " Όλα τα πειράματα ολοκληρώθηκαν επιτυχώς!"
echo "================================================="