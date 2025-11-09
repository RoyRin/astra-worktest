#!/usr/bin/env python3
"""Plot weak few-shot sweep results."""

import json
import sys
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_results(results_path: Path) -> dict:
    """Load results from JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)


def get_latest_sweep_results(results_dir: Path) -> Path:
    """Get the most recent sweep results file."""
    result_files = list(results_dir.glob("weak_sweep_*.json"))
    if not result_files:
        raise FileNotFoundError(f"No weak sweep results found in {results_dir}")

    # Sort by modification time, most recent first
    result_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return result_files[0]


def get_latest_gold_sweep_results(results_dir: Path) -> Path:
    """Get the most recent gold sweep results file."""
    result_files = list(results_dir.glob("gold_sweep_*.json"))
    if not result_files:
        raise FileNotFoundError(f"No gold sweep results found in {results_dir}")

    # Sort by modification time, most recent first
    result_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return result_files[0]


def plot_weak_sweep(results: dict, gold_results: dict = None, output_path: Path = None):
    """Plot accuracy vs number of weak few-shot examples for multiple model pairs.

    Args:
        results: Weak label sweep results
        gold_results: Optional gold label sweep results to overlay
        output_path: Path to save the plot
    """

    # Extract data for plotting, organized by weak model
    pairs_data = {}

    for pair_key, pair_info in results["model_pairs"].items():
        weak_name = pair_info["weak_model_name"]
        strong_name = pair_info["strong_model_name"]

        num_shots = []
        accuracies = []

        for result in pair_info["results"]:
            num_shots.append(result["num_few_shot"])
            accuracies.append(result["accuracy"] * 100)  # Convert to percentage

        pairs_data[pair_key] = {
            "weak_name": weak_name,
            "strong_name": strong_name,
            "num_shots": num_shots,
            "accuracies": accuracies
        }

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Define colors by weak model
    weak_model_colors = {
        "llama-3.1-8b-instruct": "#1f77b4",   # Blue
        "llama-3.1-70b-instruct": "#ff7f0e",  # Orange
    }

    # Define markers by strong model
    strong_model_markers = {
        "llama-3.1-8b-instruct": "o",     # Circle
        "llama-3.1-70b-instruct": "s",    # Square
        "qwen-2.5-72b-instruct": "^",     # Triangle
    }

    # Plot each model pair
    for pair_key, data in pairs_data.items():
        weak_name = data["weak_name"]
        strong_name = data["strong_name"]

        color = weak_model_colors.get(weak_name, "#808080")
        marker = strong_model_markers.get(strong_name, "o")
        label = f"{weak_name} → {strong_name}"

        ax.plot(
            data["num_shots"],
            data["accuracies"],
            marker=marker,
            color=color,
            linewidth=2,
            markersize=8,
            label=label,
            linestyle='-',
            alpha=0.8
        )

        # Add value labels at selected points (to avoid clutter)
        for j, (x, y) in enumerate(zip(data["num_shots"], data["accuracies"])):
            if j % 2 == 0:  # Only label every other point
                ax.annotate(
                    f'{y:.1f}%',
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha='center',
                    fontsize=7,
                    alpha=0.6
                )

    # Customize plot
    ax.set_xlabel('Number of Few-Shot Examples (Weak Labels)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Weak-to-Strong Generalization: Performance vs. Number of Weak Few-Shot Examples\nTruthfulQA Dataset',
                 fontsize=14, fontweight='bold', pad=20)

    # Set x-axis
    ax.set_xlim(-5, max(results["num_shots_list"]) + 5)
    ax.set_xticks(results["num_shots_list"])

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Create custom legend with 2 columns organized by weak model
    handles, labels = ax.get_legend_handles_labels()

    # Group pairs by weak model
    weak_8b_pairs = []
    weak_70b_pairs = []

    for pair_key, data in pairs_data.items():
        for h, l in zip(handles, labels):
            if l == f"{data['weak_name']} → {data['strong_name']}":
                if "8b" in data['weak_name']:
                    weak_8b_pairs.append((h, l))
                else:  # 70b
                    weak_70b_pairs.append((h, l))

    # Sort each group by strong model name for consistent ordering
    weak_8b_pairs.sort(key=lambda x: x[1])
    weak_70b_pairs.sort(key=lambda x: x[1])

    # Combine: first all 8b pairs, then all 70b pairs
    # This ensures 8b appears in first column with ncol=2
    sorted_items = weak_8b_pairs + weak_70b_pairs

    # Add gold baselines if provided
    if gold_results:
        # Plot gold baselines with dashed lines and lighter colors
        for model, model_data in gold_results["models"].items():
            model_name = model_data["model_name"]

            num_shots = []
            accuracies = []

            for result in model_data["results"]:
                num_shots.append(result["num_few_shot"])
                accuracies.append(result["accuracy"] * 100)

            # Use gray dashed lines for gold baselines
            line, = ax.plot(
                num_shots,
                accuracies,
                marker='D',  # Diamond marker
                color='#808080',  # Gray
                linewidth=2,
                markersize=6,
                label=f"{model_name} (gold)",
                linestyle='--',
                alpha=0.6
            )

            sorted_items.append((line, f"{model_name} (gold)"))

    sorted_handles = [item[0] for item in sorted_items]
    sorted_labels = [item[1] for item in sorted_items]

    ax.legend(sorted_handles, sorted_labels, loc='lower right', fontsize=10,
              framealpha=0.9, ncol=2)

    # Add info text
    test_size = results["dataset"]["test_size"]
    timestamp = results["timestamp"]

    info_text = f"Test size: {test_size} questions\nRun: {timestamp}"

    ax.text(0.02, 0.98, info_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()

    return fig, ax


def print_summary(results: dict):
    """Print summary table of results."""
    print("\n" + "=" * 70)
    print("WEAK FEW-SHOT SWEEP RESULTS")
    print("=" * 70)
    print(f"Dataset: {results['dataset']['name']}")
    print(f"Test size: {results['dataset']['test_size']}")
    print(f"Timestamp: {results['timestamp']}")
    print(f"\nNumber of shots tested: {results['num_shots_list']}")

    print("\n" + "-" * 70)
    print("Accuracy by Model Pair and Shot Count:")
    print("-" * 70)

    # Create table header
    shot_counts = results['num_shots_list']
    header = "Model Pair".ljust(40) + " | " + " | ".join(f"{n:3d}".center(6) for n in shot_counts)
    print(header)
    print("-" * len(header))

    # Print each pair's results
    for pair_key, pair_info in results["model_pairs"].items():
        weak_name = pair_info["weak_model_name"]
        strong_name = pair_info["strong_model_name"]
        pair_label = f"{weak_name} → {strong_name}"

        row = pair_label.ljust(40) + " | "

        accuracies = []
        for result in pair_info["results"]:
            acc = result["accuracy"] * 100
            accuracies.append(f"{acc:.1f}%".center(6))

        row += " | ".join(accuracies)
        print(row)

    print("=" * 70 + "\n")


def main():
    """Main function."""

    # Get results directory
    # Go up to code/, then up to project root, then to results/
    results_dir = Path(__file__).parent.parent.parent / "results"

    # Determine which results file to use
    if len(sys.argv) > 1:
        # Use provided path
        results_path = Path(sys.argv[1])
        if not results_path.exists():
            print(f"Error: File not found: {results_path}")
            sys.exit(1)
    else:
        # Use latest results
        try:
            results_path = get_latest_sweep_results(results_dir)
            print(f"Using latest results: {results_path.name}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)

    # Load results
    print(f"Loading results from: {results_path}")
    results = load_results(results_path)

    # Load gold baseline results
    gold_results = None
    try:
        gold_results_path = get_latest_gold_sweep_results(results_dir)
        print(f"Loading gold baselines from: {gold_results_path.name}")
        gold_results = load_results(gold_results_path)
    except FileNotFoundError:
        print("Warning: No gold sweep results found, skipping gold baselines")

    # Print summary
    print_summary(results)

    # Create output path with timestamp
    plot_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = results_path.parent / f"{results_path.stem}_plot_{plot_timestamp}.png"

    # Plot results
    plot_weak_sweep(results, gold_results, output_path)


if __name__ == "__main__":
    main()
