#!/usr/bin/env python3
"""Plot baseline results from JSON files."""

import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_results(results_path: Path) -> dict:
    """Load results from JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)


def get_latest_results(results_dir: Path) -> Path:
    """Get the most recent results file."""
    result_files = list(results_dir.glob("baseline_*.json"))
    if not result_files:
        raise FileNotFoundError(f"No baseline results found in {results_dir}")

    # Sort by modification time, most recent first
    result_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return result_files[0]


def plot_baseline_results(results: dict, output_path: Path = None):
    """Plot baseline accuracies as a bar chart."""

    # Extract model names and accuracies
    models = []
    accuracies = []

    for model_id, model_results in results["models"].items():
        model_name = model_results["model_name"]
        accuracy = model_results["accuracy"]
        models.append(model_name)
        accuracies.append(accuracy * 100)  # Convert to percentage

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create bar chart
    x = np.arange(len(models))
    bars = ax.bar(x, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(models)])

    # Customize plot
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'TruthfulQA Baseline Accuracies (No Few-Shot)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%',
                ha='center', va='bottom', fontweight='bold')

    # Add grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Add info text
    num_shots = results["few_shot"]["num_examples"]
    test_size = results["dataset"]["test_size"]
    timestamp = results["timestamp"]

    info_text = f"Test size: {test_size} questions"
    if num_shots > 0:
        info_text += f"\nFew-shot: {num_shots} examples"
    else:
        info_text += f"\nFew-shot: None (zero-shot)"
    info_text += f"\nRun: {timestamp}"

    ax.text(0.02, 0.98, info_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # Set y-axis limits
    ax.set_ylim(0, max(accuracies) * 1.15)

    plt.tight_layout()

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()

    return fig, ax


def main():
    """Main function."""

    # Get results directory
    results_dir = Path(__file__).parent.parent / "results"

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
            results_path = get_latest_results(results_dir)
            print(f"Using latest results: {results_path.name}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)

    # Load results
    print(f"Loading results from: {results_path}")
    results = load_results(results_path)

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Dataset: {results['dataset']['name']}")
    print(f"Test size: {results['dataset']['test_size']}")
    print(f"Few-shot examples: {results['few_shot']['num_examples']}")
    print(f"Timestamp: {results['timestamp']}")
    print("\nModel Accuracies:")
    for model_id, model_results in results["models"].items():
        model_name = model_results["model_name"]
        accuracy = model_results["accuracy"]
        num_correct = model_results["num_correct"]
        num_total = model_results["num_total"]
        print(f"  {model_name}: {accuracy:.1%} ({num_correct}/{num_total})")
    print("=" * 70 + "\n")

    # Create output path
    output_path = results_path.parent / f"{results_path.stem}_plot.png"

    # Plot results
    plot_baseline_results(results, output_path)


if __name__ == "__main__":
    main()
