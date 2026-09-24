#!/usr/bin/env python3

import argparse
import os
import numpy as np

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic data in chunks to avoid OOM errors."
    )

    parser.add_argument("--samples", type=int, required=True,
                        help="Number of samples/rows N")
    parser.add_argument("--features", type=int, required=True,
                        help="Number of features/columns D")
    parser.add_argument("--output", type=str, required=True,
                        help="Output binary file")
    parser.add_argument("--dtype", type=str, default="float64",
                        choices=["float32", "float64"],
                        help="Data type for output array")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--chunk-size", type=int, default=100000,
                        help="Number of rows per chunk to save memory")

    args = parser.parse_args()

    print("Generating data (Out-of-Core/Chunked)...")
    print(f"N = {args.samples}")
    print(f"D = {args.features}")
    print(f"dtype = {args.dtype}")
    print(f"Chunk size = {args.chunk_size} rows")

    np.random.seed(args.seed)
    
    # Διαγραφή του αρχείου αν υπάρχει ήδη, για να κάνουμε σωστό append (ab)
    if os.path.exists(args.output):
        os.remove(args.output)

    rows_generated = 0
    with open(args.output, "ab") as f_out:
        while rows_generated < args.samples:
            current_chunk = min(args.chunk_size, args.samples - rows_generated)
            
            # Δημιουργία τυχαίων δεδομένων με κανονική κατανομή
            # Πολλαπλασιάζουμε με τυχαίες κλίμακες/μετατοπίσεις ανά στήλη
            # ώστε να έχουν νόημα οι scalers (standard & minmax)
            X_chunk = np.random.randn(current_chunk, args.features)
            
            # Τυχαία κλιμάκωση για να έχουν διαφορετικό variance/mean οι στήλες
            scales = np.random.uniform(1.0, 100.0, size=args.features)
            shifts = np.random.uniform(-50.0, 50.0, size=args.features)
            X_chunk = X_chunk * scales + shifts

            if args.dtype == "float32":
                X_chunk = X_chunk.astype(np.float32)
            else:
                X_chunk = X_chunk.astype(np.float64)

            # Εγγραφή απευθείας στο αρχείο
            X_chunk.tofile(f_out)
            
            rows_generated += current_chunk
            
            # Προαιρετική εκτύπωση προόδου αν τα δεδομένα είναι πάρα πολλά
            if rows_generated % (args.chunk_size * 10) == 0:
                print(f"Progress: {rows_generated} / {args.samples} rows generated...")

    print("\nDataset generated successfully.")
    print(f"Output file: {args.output}")
    file_size_bytes = os.path.getsize(args.output)
    print(f"File size: {file_size_bytes} bytes ({file_size_bytes / (1024 ** 3):.3f} GB)")

if __name__ == "__main__":
    main()