#!/bin/bash

SCRIPTS_DIR="$(dirname "$0")/scripts"

for py_file in "$SCRIPTS_DIR"/*.py; do
    base="${py_file%.py}"
    output_file="${base}_output.txt"

    echo "Running $(basename "$py_file")..."
    python3 "$py_file" > "$output_file"

    if [[ $? -eq 0 ]]; then
        echo "  ✓ Output written to $(basename "$output_file")"
    else
        echo "  ✗ Error running $(basename "$py_file")"
    fi
done

echo "Done."