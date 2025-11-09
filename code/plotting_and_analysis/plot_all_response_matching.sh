#!/bin/bash
# Plot response matching for all weak-to-strong model pairs

set -e  # Exit on error

echo "========================================================================"
echo "Plotting Response Matching for All W2S Pairs"
echo "========================================================================"

# Define model pairs (same as in sweep_number_of_weak_labels.py)
# Format: "weak_model|strong_model"
pairs=(
    "meta-llama/llama-3.1-8b-instruct|meta-llama/llama-3.1-8b-instruct"
    "meta-llama/llama-3.1-8b-instruct|meta-llama/llama-3.1-70b-instruct"
    "meta-llama/llama-3.1-8b-instruct|qwen/qwen-2.5-72b-instruct"
    "meta-llama/llama-3.1-70b-instruct|meta-llama/llama-3.1-70b-instruct"
    "meta-llama/llama-3.1-70b-instruct|qwen/qwen-2.5-72b-instruct"
    "qwen/qwen-2.5-72b-instruct|qwen/qwen-2.5-72b-instruct"
)

echo ""
echo "Found ${#pairs[@]} model pairs to process"
echo ""

# Counter
count=0
total=${#pairs[@]}

# Process each pair
for pair in "${pairs[@]}"; do
    ((count++))

    # Split on |
    IFS='|' read -r weak strong <<< "$pair"

    echo "[$count/$total] Processing: $weak → $strong"

    # Run the plotting script
    python code/plotting_and_analysis/plot_response_matching.py \
        --weak-model "$weak" \
        --strong-model "$strong"

    echo ""
done

echo "========================================================================"
echo "All plots completed!"
echo "Plots saved to: results/"
echo "========================================================================"
