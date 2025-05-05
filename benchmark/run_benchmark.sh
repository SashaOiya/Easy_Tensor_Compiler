#!/bin/bash

INPUT_DIR="../benchmark/data"
if [ ! -d "$INPUT_DIR" ]; then
    echo "❌ Error: Input directory '$INPUT_DIR' does not exist."
    exit 1
fi
OUTPUT_DIR="./results"
BINARY="./benchmark/matmul_benchmark"
if [ ! -x "$BINARY" ]; then
    echo "❌ Error: Benchmark binary '$BINARY' not found or not executable."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

i=1
for input_file in "$INPUT_DIR"/data*.in; do
    echo "Running benchmark with $input_file"
    "$BINARY" < "$input_file" --benchmark_out="$OUTPUT_DIR/result${i}.json" --benchmark_out_format=json
    i=$((i + 1))
done

echo "All benchmarks completed. Results saved in $OUTPUT_DIR."
