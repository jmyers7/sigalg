#!/bin/bash

SCRIPTS_DIR="$(dirname "$0")/scripts"

# Scripts that generate plots instead of text output
SKIP_OUTPUT=("polynomial_regression.py" "random_walk_diffusion.py" "fourier_polynomials.py")

for py_file in "$SCRIPTS_DIR"/*.py; do
    base="${py_file%.py}"
    basename_file=$(basename "$py_file")
    output_file="${base}_output.txt"

    echo "Running $basename_file..."
    
    # Change to scripts directory before running
    cd "$SCRIPTS_DIR"
    
    # Check if this script should be skipped for output capture
    if [[ " ${SKIP_OUTPUT[@]} " =~ " ${basename_file} " ]]; then
        MPLBACKEND=Agg python3 "$basename_file"
        if [[ $? -eq 0 ]]; then
            echo "  ✓ Executed (no output capture)"
        else
            echo "  ✗ Error running $basename_file"
        fi
    else
        python3 "$basename_file" > "${basename_file%.py}_output.txt"
        if [[ $? -eq 0 ]]; then
            echo "  ✓ Output written to ${basename_file%.py}_output.txt"
        else
            echo "  ✗ Error running $basename_file"
        fi
    fi
    
    # Change back to original directory
    cd - > /dev/null
done

echo "Done."