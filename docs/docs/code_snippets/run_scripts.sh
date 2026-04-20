#!/bin/bash

export MPLBACKEND=Agg

SCRIPTS_DIR="$(dirname "$0")/scripts"

# Scripts that generate plots instead of text output
SKIP_OUTPUT=("polynomial_regression.py" "fourier_polynomials.py" "random_walk_diffusion.py" "asian_option_pricing.py")

for py_file in "$SCRIPTS_DIR"/*.py; do
    base="${py_file%.py}"
    basename_file=$(basename "$py_file")

    echo "Running $basename_file..."
    
    cd "$SCRIPTS_DIR"
    
    if [[ " ${SKIP_OUTPUT[@]} " =~ " ${basename_file} " ]]; then
        python3 "$basename_file"
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
    
    cd - > /dev/null
done

echo "Done."