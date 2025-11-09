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

    # Define markers by model (used for both strong models and gold baselines)
    model_markers = {
        "llama-3.1-8b-instruct": "o",     # Circle
        "llama-3.1-70b-instruct": "s",    # Square
        "qwen-2.5-72b-instruct": "^",     # Triangle
    }

    # Plot each model pair
    for pair_key, data in pairs_data.items():
        weak_name = data["weak_name"]
        strong_name = data["strong_name"]

        color = weak_model_colors.get(weak_name, "#808080")
        marker = model_markers.get(strong_name, "o")
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

    # Create custom legend with 3 columns organized by weak model
    handles, labels = ax.get_legend_handles_labels()

    # Group pairs by weak model
    weak_8b_pairs = []
    weak_70b_pairs = []
    gold_pairs = []

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

    # Add gold baselines if provided
    if gold_results:
        # Plot gold baselines with dashed lines and lighter colors
        # Sort models to ensure consistent ordering
        sorted_models = sorted(gold_results["models"].items(),
                              key=lambda x: x[1]["model_name"])

        for model, model_data in sorted_models:
            model_name = model_data["model_name"]

            num_shots = []
            accuracies = []

            for result in model_data["results"]:
                num_shots.append(result["num_few_shot"])
                accuracies.append(result["accuracy"] * 100)

            # Use gray dashed lines for gold baselines with same markers as model
            marker = model_markers.get(model_name, "o")

            line, = ax.plot(
                num_shots,
                accuracies,
                marker=marker,
                color='#808080',  # Gray
                linewidth=2,
                markersize=8,
                label=f"{model_name} (gold)",
                linestyle='--',
                alpha=0.6
            )

            gold_pairs.append((line, f"{model_name} (gold)"))

    # Combine: first gold pairs, then 8b pairs, then 70b pairs
    # This ensures gold appears in first column with ncol=3
    sorted_items = gold_pairs + weak_8b_pairs + weak_70b_pairs

    sorted_handles = [item[0] for item in sorted_items]
    sorted_labels = [item[1] for item in sorted_items]

    ax.legend(sorted_handles, sorted_labels, loc='lower right', fontsize=10,
              framealpha=0.9, ncol=3)

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


def get_baseline_accuracy(baseline_results_dir: Path, model_name: str) -> float:
    """Get baseline accuracy for a model (zero-shot baseline).

    Args:
        baseline_results_dir: Directory containing baseline results
        model_name: Name of the model

    Returns:
        Baseline accuracy as a percentage
    """
    model_slug = model_name.replace('/', '_')
    baseline_file = baseline_results_dir / f"{model_slug}_baseline.json"

    if not baseline_file.exists():
        raise FileNotFoundError(f"Baseline not found: {baseline_file}")

    with open(baseline_file, 'r') as f:
        baseline_data = json.load(f)

    # Get zero-shot accuracy (baseline with no few-shot examples)
    return baseline_data["results"]["accuracy"] * 100


def plot_pgr(results: dict, baseline_results_dir: Path, output_path: Path = None):
    """Plot Performance Gap Recovered (PGR) vs number of weak few-shot examples.

    PGR = (W2S - Weak) / (Strong - Weak)

    Args:
        results: Weak label sweep results
        baseline_results_dir: Directory containing baseline results
        output_path: Path to save the plot
    """

    # Load baseline accuracies for all models
    baselines = {}
    all_models = set()

    for pair_key, pair_info in results["model_pairs"].items():
        all_models.add(pair_info["weak_model"])
        all_models.add(pair_info["strong_model"])

    print("\nLoading baseline accuracies...")
    for model in all_models:
        try:
            baselines[model] = get_baseline_accuracy(baseline_results_dir, model)
            print(f"  {model}: {baselines[model]:.2f}%")
        except FileNotFoundError as e:
            print(f"  Warning: {e}")
            return

    # Calculate PGR for each pair
    pairs_pgr = {}

    for pair_key, pair_info in results["model_pairs"].items():
        weak_model = pair_info["weak_model"]
        strong_model = pair_info["strong_model"]
        weak_name = pair_info["weak_model_name"]
        strong_name = pair_info["strong_model_name"]

        weak_baseline = baselines[weak_model]
        strong_baseline = baselines[strong_model]

        # Skip if weak == strong (self-supervision) or if gap is too small
        if weak_model == strong_model or abs(strong_baseline - weak_baseline) < 0.1:
            continue

        num_shots = []
        pgr_values = []

        for result in pair_info["results"]:
            w2s_acc = result["accuracy"] * 100

            # PGR = (W2S - Weak) / (Strong - Weak)
            pgr = (w2s_acc - weak_baseline) / (strong_baseline - weak_baseline) * 100

            num_shots.append(result["num_few_shot"])
            pgr_values.append(pgr)

        pairs_pgr[pair_key] = {
            "weak_name": weak_name,
            "strong_name": strong_name,
            "num_shots": num_shots,
            "pgr_values": pgr_values,
            "weak_baseline": weak_baseline,
            "strong_baseline": strong_baseline
        }

    if not pairs_pgr:
        print("No valid pairs for PGR calculation (all are self-supervision or zero gap)")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Define colors by weak model
    weak_model_colors = {
        "llama-3.1-8b-instruct": "#1f77b4",   # Blue
        "llama-3.1-70b-instruct": "#ff7f0e",  # Orange
    }

    # Define markers by strong model
    model_markers = {
        "llama-3.1-8b-instruct": "o",     # Circle
        "llama-3.1-70b-instruct": "s",    # Square
        "qwen-2.5-72b-instruct": "^",     # Triangle
    }

    # Plot each model pair
    for pair_key, data in pairs_pgr.items():
        weak_name = data["weak_name"]
        strong_name = data["strong_name"]

        color = weak_model_colors.get(weak_name, "#808080")
        marker = model_markers.get(strong_name, "o")
        label = f"{weak_name} → {strong_name}"

        ax.plot(
            data["num_shots"],
            data["pgr_values"],
            marker=marker,
            color=color,
            linewidth=2,
            markersize=8,
            label=label,
            linestyle='-',
            alpha=0.8
        )

        # Add value labels at selected points
        for j, (x, y) in enumerate(zip(data["num_shots"], data["pgr_values"])):
            if j % 2 == 0:  # Only label every other point
                ax.annotate(
                    f'{y:.0f}%',
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha='center',
                    fontsize=7,
                    alpha=0.6
                )

    # Add horizontal lines at 0% (weak baseline) and 100% (strong baseline)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='Weak baseline (0%)')
    ax.axhline(y=100, color='green', linestyle='--', linewidth=1.5, alpha=0.5, label='Strong baseline (100%)')

    # Customize plot
    ax.set_xlabel('Number of Few-Shot Examples (Weak Labels)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Performance Gap Recovered (%)', fontsize=12, fontweight='bold')
    ax.set_title('Weak-to-Strong Generalization: Performance Gap Recovered (PGR)\nPGR = (W2S - Weak) / (Strong - Weak) × 100%',
                 fontsize=14, fontweight='bold', pad=20)

    # Set x-axis
    ax.set_xlim(-5, max(results["num_shots_list"]) + 5)
    ax.set_xticks(results["num_shots_list"])

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Add legend
    ax.legend(loc='best', fontsize=10, framealpha=0.9, ncol=2)

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
        print(f"PGR plot saved to: {output_path}")
    else:
        plt.show()

    return fig, ax


def plot_pgr_with_gold_ceiling(results: dict, baseline_results_dir: Path, gold_results: dict, output_path: Path = None):
    """Plot Performance Gap Recovered (PGR) with gold-label ceiling vs number of weak few-shot examples.

    PGR = (W2S - Weak) / (Strong_gold(N) - Weak)

    Where Strong_gold(N) is the strong model's accuracy with N gold labels (same N as weak labels).

    Args:
        results: Weak label sweep results
        baseline_results_dir: Directory containing baseline results
        gold_results: Gold label sweep results
        output_path: Path to save the plot
    """

    if gold_results is None:
        print("Warning: No gold results provided, cannot generate PGR with gold ceiling plot")
        return

    # Load baseline accuracies for all models
    baselines = {}
    all_models = set()

    for pair_key, pair_info in results["model_pairs"].items():
        all_models.add(pair_info["weak_model"])
        all_models.add(pair_info["strong_model"])

    print("\nLoading baseline accuracies for PGR (gold ceiling)...")
    for model in all_models:
        try:
            baselines[model] = get_baseline_accuracy(baseline_results_dir, model)
            print(f"  {model}: {baselines[model]:.2f}%")
        except FileNotFoundError as e:
            print(f"  Warning: {e}")
            return

    # Build gold accuracy lookup: model -> num_shots -> accuracy
    gold_accuracies = {}
    for model, model_data in gold_results["models"].items():
        model_name = model_data["model_name"]
        gold_accuracies[model] = {}
        for result in model_data["results"]:
            num_shots = result["num_few_shot"]
            acc = result["accuracy"] * 100
            gold_accuracies[model][num_shots] = acc

    # Calculate PGR for each pair
    pairs_pgr = {}

    for pair_key, pair_info in results["model_pairs"].items():
        weak_model = pair_info["weak_model"]
        strong_model = pair_info["strong_model"]
        weak_name = pair_info["weak_model_name"]
        strong_name = pair_info["strong_model_name"]

        weak_baseline = baselines[weak_model]

        # Skip if weak == strong (self-supervision)
        if weak_model == strong_model:
            continue

        # Check if we have gold results for the strong model
        if strong_model not in gold_accuracies:
            print(f"  Warning: No gold results for {strong_model}, skipping")
            continue

        num_shots = []
        pgr_values = []

        for result in pair_info["results"]:
            w2s_acc = result["accuracy"] * 100
            num_few_shot = result["num_few_shot"]

            # Get strong model's accuracy with N gold labels
            if num_few_shot not in gold_accuracies[strong_model]:
                print(f"  Warning: No gold result for {strong_model} at {num_few_shot} shots, skipping")
                continue

            strong_gold_ceiling = gold_accuracies[strong_model][num_few_shot]

            # Skip if gap is too small
            if abs(strong_gold_ceiling - weak_baseline) < 0.1:
                continue

            # PGR = (W2S - Weak) / (Strong_gold(N) - Weak)
            pgr = (w2s_acc - weak_baseline) / (strong_gold_ceiling - weak_baseline) * 100

            num_shots.append(num_few_shot)
            pgr_values.append(pgr)

        if len(num_shots) > 0:
            pairs_pgr[pair_key] = {
                "weak_name": weak_name,
                "strong_name": strong_name,
                "num_shots": num_shots,
                "pgr_values": pgr_values,
                "weak_baseline": weak_baseline,
            }

    if not pairs_pgr:
        print("No valid pairs for PGR calculation with gold ceiling")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Define colors by weak model
    weak_model_colors = {
        "llama-3.1-8b-instruct": "#1f77b4",   # Blue
        "llama-3.1-70b-instruct": "#ff7f0e",  # Orange
    }

    # Define markers by strong model
    model_markers = {
        "llama-3.1-8b-instruct": "o",     # Circle
        "llama-3.1-70b-instruct": "s",    # Square
        "qwen-2.5-72b-instruct": "^",     # Triangle
    }

    # Plot each model pair
    for pair_key, data in pairs_pgr.items():
        weak_name = data["weak_name"]
        strong_name = data["strong_name"]

        color = weak_model_colors.get(weak_name, "#808080")
        marker = model_markers.get(strong_name, "o")
        label = f"{weak_name} → {strong_name}"

        ax.plot(
            data["num_shots"],
            data["pgr_values"],
            marker=marker,
            color=color,
            linewidth=2,
            markersize=8,
            label=label,
            linestyle='-',
            alpha=0.8
        )

        # Add value labels at selected points
        for j, (x, y) in enumerate(zip(data["num_shots"], data["pgr_values"])):
            if j % 2 == 0:  # Only label every other point
                ax.annotate(
                    f'{y:.0f}%',
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha='center',
                    fontsize=7,
                    alpha=0.6
                )

    # Add horizontal lines at 0% (weak baseline) and 100% (strong gold ceiling)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='Weak baseline (0%)')
    ax.axhline(y=100, color='green', linestyle='--', linewidth=1.5, alpha=0.5, label='Strong w/ N gold labels (100%)')

    # Customize plot
    ax.set_xlabel('Number of Few-Shot Examples (Weak Labels)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Performance Gap Recovered (%)', fontsize=12, fontweight='bold')
    ax.set_title('Weak-to-Strong Generalization: PGR with Gold-Label Ceiling\nPGR = (W2S - Weak) / (Strong_gold(N) - Weak) × 100%',
                 fontsize=14, fontweight='bold', pad=20)

    # Set x-axis
    ax.set_xlim(-5, max(results["num_shots_list"]) + 5)
    ax.set_xticks(results["num_shots_list"])

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Add legend
    ax.legend(loc='best', fontsize=10, framealpha=0.9, ncol=2)

    # Add info text
    test_size = results["dataset"]["test_size"]
    timestamp = results["timestamp"]

    info_text = f"Test size: {test_size} questions\nRun: {timestamp}\nCeiling: Strong model with N gold labels"

    ax.text(0.02, 0.98, info_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"PGR (gold ceiling) plot saved to: {output_path}")
    else:
        plt.show()

    return fig, ax


def plot_self_supervision(results: dict, gold_results: dict = None, output_path: Path = None):
    """Plot self-supervision cases where weak model = strong model.

    Args:
        results: Weak label sweep results
        gold_results: Optional gold label sweep results to overlay
        output_path: Path to save the plot
    """

    # Extract only self-supervision pairs (weak == strong)
    self_sup_data = {}

    for pair_key, pair_info in results["model_pairs"].items():
        weak_name = pair_info["weak_model_name"]
        strong_name = pair_info["strong_model_name"]

        # Only include if weak == strong
        if weak_name == strong_name:
            num_shots = []
            accuracies = []

            for result in pair_info["results"]:
                num_shots.append(result["num_few_shot"])
                accuracies.append(result["accuracy"] * 100)

            self_sup_data[weak_name] = {
                "num_shots": num_shots,
                "accuracies": accuracies
            }

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Define colors for self-supervision models
    self_sup_colors = {
        "llama-3.1-8b-instruct": "#1f77b4",   # Blue
        "llama-3.1-70b-instruct": "#ff7f0e",  # Orange
    }

    # Plot each self-supervision case
    for model_name, data in sorted(self_sup_data.items()):
        color = self_sup_colors.get(model_name, "#2ca02c")

        ax.plot(
            data["num_shots"],
            data["accuracies"],
            marker='o',
            color=color,
            linewidth=2.5,
            markersize=10,
            label=f"{model_name} (self-sup)",
            linestyle='-',
            alpha=0.8
        )

        # Add value labels at all points
        for x, y in zip(data["num_shots"], data["accuracies"]):
            ax.annotate(
                f'{y:.1f}%',
                (x, y),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                fontsize=9,
                alpha=0.7
            )

    # Define markers for consistency
    model_markers = {
        "llama-3.1-8b-instruct": "o",     # Circle
        "llama-3.1-70b-instruct": "s",    # Square
        "qwen-2.5-72b-instruct": "^",     # Triangle
    }

    # Add gold baselines if provided
    if gold_results:
        sorted_models = sorted(gold_results["models"].items(),
                              key=lambda x: x[1]["model_name"])

        for model, model_data in sorted_models:
            model_name = model_data["model_name"]

            num_shots = []
            accuracies = []

            for result in model_data["results"]:
                num_shots.append(result["num_few_shot"])
                accuracies.append(result["accuracy"] * 100)

            # Use gray dashed lines for gold baselines with same markers as model
            marker = model_markers.get(model_name, "o")

            ax.plot(
                num_shots,
                accuracies,
                marker=marker,
                color='#808080',
                linewidth=2,
                markersize=9,
                label=f"{model_name} (gold)",
                linestyle='--',
                alpha=0.6
            )

    # Customize plot
    ax.set_xlabel('Number of Few-Shot Examples', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Self-Supervision: Models Learning from Their Own Predictions\nTruthfulQA Dataset',
                 fontsize=15, fontweight='bold', pad=20)

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

    info_text = f"Test size: {test_size} questions\nRun: {timestamp}\nSelf-supervision: weak = strong"

    ax.text(0.02, 0.98, info_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Self-supervision plot saved to: {output_path}")
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

    # Create plots directory
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Create output paths with timestamp
    plot_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path_full = plots_dir / f"{results_path.stem}_plot_{plot_timestamp}.png"
    output_path_self_sup = plots_dir / f"{results_path.stem}_plot_self_sup_{plot_timestamp}.png"
    output_path_pgr = plots_dir / f"{results_path.stem}_plot_pgr_{plot_timestamp}.png"
    output_path_pgr_gold = plots_dir / f"{results_path.stem}_plot_pgr_gold_ceiling_{plot_timestamp}.png"

    # Plot full results
    print("\nGenerating full plot...")
    plot_weak_sweep(results, gold_results, output_path_full)

    # Plot self-supervision only
    print("\nGenerating self-supervision plot...")
    plot_self_supervision(results, gold_results, output_path_self_sup)

    # Plot PGR (Performance Gap Recovered) - zero-shot ceiling
    print("\nGenerating PGR plot (zero-shot ceiling)...")
    baseline_results_dir = results_dir / "weak_labels"
    plot_pgr(results, baseline_results_dir, output_path_pgr)

    # Plot PGR with gold-label ceiling
    print("\nGenerating PGR plot (gold-label ceiling)...")
    plot_pgr_with_gold_ceiling(results, baseline_results_dir, gold_results, output_path_pgr_gold)


if __name__ == "__main__":
    main()
