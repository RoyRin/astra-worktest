#!/usr/bin/env python3
"""Plot gold few-shot sweep results."""

import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_results(results_path: Path) -> dict:
    """Load results from JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)


def get_latest_sweep_results(results_dir: Path) -> Path:
    """Get the most recent sweep results file."""
    result_files = list(results_dir.glob("gold_sweep_*.json"))
    if not result_files:
        raise FileNotFoundError(f"No gold sweep results found in {results_dir}")

    # Sort by modification time, most recent first
    result_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return result_files[0]


def plot_gold_sweep(results: dict, output_path: Path = None):
    """Plot accuracy vs number of few-shot examples for multiple models."""

    # Extract data for plotting
    models_data = {}

    for model_id, model_info in results["models"].items():
        model_name = model_info["model_name"]
        num_shots = []
        accuracies = []

        for result in model_info["results"]:
            num_shots.append(result["num_few_shot"])
            accuracies.append(result["accuracy"] * 100)  # Convert to percentage

        models_data[model_name] = {
            "num_shots": num_shots,
            "accuracies": accuracies
        }

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    # Define colors and markers for each model
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    markers = ['o', 's', '^']

    # Plot each model
    for i, (model_name, data) in enumerate(models_data.items()):
        ax.plot(
            data["num_shots"],
            data["accuracies"],
            marker=markers[i],
            color=colors[i],
            linewidth=2,
            markersize=8,
            label=model_name,
            linestyle='-',
            alpha=0.8
        )

        # Add value labels at each point
        for x, y in zip(data["num_shots"], data["accuracies"]):
            ax.annotate(
                f'{y:.1f}%',
                (x, y),
                textcoords="offset points",
                xytext=(0, 8),
                ha='center',
                fontsize=8,
                alpha=0.7
            )

    # Customize plot
    ax.set_xlabel('Number of Few-Shot Examples (Gold Labels)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance vs. Number of Gold Few-Shot Examples\nTruthfulQA Dataset',
                 fontsize=14, fontweight='bold', pad=20)

    # Set x-axis
    ax.set_xlim(-5, max(results["num_shots_list"]) + 5)
    ax.set_xticks(results["num_shots_list"])

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Add legend
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)

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
    print("GOLD FEW-SHOT SWEEP RESULTS")
    print("=" * 70)
    print(f"Dataset: {results['dataset']['name']}")
    print(f"Test size: {results['dataset']['test_size']}")
    print(f"Timestamp: {results['timestamp']}")
    print(f"\nNumber of shots tested: {results['num_shots_list']}")

    print("\n" + "-" * 70)
    print("Accuracy by Model and Shot Count:")
    print("-" * 70)

    # Create table header
    shot_counts = results['num_shots_list']
    header = "Model".ljust(30) + " | " + " | ".join(f"{n:3d}".center(6) for n in shot_counts)
    print(header)
    print("-" * len(header))

    # Print each model's results
    for model_id, model_info in results["models"].items():
        model_name = model_info["model_name"]
        row = model_name.ljust(30) + " | "

        accuracies = []
        for result in model_info["results"]:
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

    # Print summary
    print_summary(results)

    # Create output path
    output_path = results_path.parent / f"{results_path.stem}_plot.png"

    # Plot results
    plot_gold_sweep(results, output_path)


if __name__ == "__main__":
    main()
