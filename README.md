## Δομή Project
- `serial_scaler.c`: Σειριακή υλοποίηση (Reference)
- `simd_scaler.c`: Υλοποίηση με Vector Instructions (AVX2)
- `omp_scaler.c`: Υλοποίηση shared-memory με OpenMP
- `mpi_scaler.c`: Υλοποίηση distributed-memory με κλασικό Parallel File I/O (MPI)
- `cuda_scaler.cu`: Υλοποίηση με επιτάχυνση GPU μέσω CUDA
- `generate_data.py`: Παραγωγή synthetic datasets (Out-of-Core)
- `check_correctness.py`: Έλεγχος λαθών σε σχέση με το scikit-learn (Out-of-Core)
- `Makefile`: Script μεταγλώττισης
- `run_experiments.sh`: Αυτοματοποιημένη εκτέλεση πειραμάτων

## Μεταγλώττιση (Compile)
Για τη μεταγλώττιση όλων των εκδόσεων, εκτελέστε:
make clean
make 

## Εκτέλεση
Για να τρέξετε όλα τα πειράματα (παραγωγή αρχείων, εκτέλεση scalers για όλα τα μεγέθη και έλεγχος ορθότητας) εκτελέστε:
./run_experiments.sh