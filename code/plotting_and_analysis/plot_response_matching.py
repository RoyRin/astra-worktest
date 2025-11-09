#!/usr/bin/env python3
"""Plot response matching patterns as a function of number of few-shot examples.

For a single weak-to-strong pair, shows how W2S responses align with weak/strong
models as the number of weak labels increases.
"""

import json
import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

# Add parent directory to path for imports (go up to code/)
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset_utils import load_truthful_qa


def load_results(results_path: Path) -> dict:
    """Load results from JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)


def get_baseline_results(weak_labels_dir: Path, model: str) -> dict:
    """Get baseline results for a model."""
    model_slug = model.replace('/', '_')
    baseline_file = weak_labels_dir / f"{model_slug}_baseline.json"

    if not baseline_file.exists():
        raise FileNotFoundError(f"Baseline not found: {baseline_file}")

    return load_results(baseline_file)


def analyze_response_matching_over_shots(
    w2s_results: dict,
    weak_baseline: dict,
    strong_baseline: dict,
    ground_truth: list
) -> tuple:
    """Analyze response matching for all shot counts.

    Args:
        w2s_results: W2S results data with multiple shot counts
        weak_baseline: Weak model baseline results
        strong_baseline: Strong model baseline results
        ground_truth: List of ground truth answers

    Returns:
        Tuple of (all_results, weak_wrong_results) for each shot count
    """
    weak_preds = weak_baseline["results"]["predictions"]
    strong_preds = strong_baseline["results"]["predictions"]

    results_by_shots = []
    weak_wrong_results = []

    for shot_result in w2s_results["results_by_shot_count"]:
        num_shots = shot_result["num_few_shot"]
        w2s_preds = shot_result["predictions"]

        num_questions = len(w2s_preds)

        # Count matching patterns (all questions)
        matches_weak = 0
        matches_strong = 0
        matches_both = 0
        matches_neither = 0

        # Count matching patterns (only when weak is wrong)
        weak_wrong_count = 0
        ww_matches_weak = 0
        ww_matches_strong = 0
        ww_matches_both = 0
        ww_matches_neither = 0

        for i in range(num_questions):
            w_pred = weak_preds[i]
            s_pred = strong_preds[i]
            w2s_pred = w2s_preds[i]

            is_match_weak = (w2s_pred == w_pred)
            is_match_strong = (w2s_pred == s_pred)

            # All questions
            if is_match_weak and is_match_strong:
                matches_both += 1
            elif is_match_weak:
                matches_weak += 1
            elif is_match_strong:
                matches_strong += 1
            else:
                matches_neither += 1

            # Only when weak is wrong
            if weak_preds[i] != ground_truth[i]:
                weak_wrong_count += 1
                if is_match_weak and is_match_strong:
                    ww_matches_both += 1
                elif is_match_weak:
                    ww_matches_weak += 1
                elif is_match_strong:
                    ww_matches_strong += 1
                else:
                    ww_matches_neither += 1

        results_by_shots.append({
            "num_shots": num_shots,
            "matches_weak": matches_weak / num_questions * 100,
            "matches_strong": matches_strong / num_questions * 100,
            "matches_both": matches_both / num_questions * 100,
            "matches_neither": matches_neither / num_questions * 100,
        })

        if weak_wrong_count > 0:
            weak_wrong_results.append({
                "num_shots": num_shots,
                "matches_weak": ww_matches_weak / weak_wrong_count * 100,
                "matches_strong": ww_matches_strong / weak_wrong_count * 100,
                "matches_both": ww_matches_both / weak_wrong_count * 100,
                "matches_neither": ww_matches_neither / weak_wrong_count * 100,
                "weak_wrong_count": weak_wrong_count,
            })
        else:
            weak_wrong_results.append({
                "num_shots": num_shots,
                "matches_weak": 0,
                "matches_strong": 0,
                "matches_both": 0,
                "matches_neither": 0,
                "weak_wrong_count": 0,
            })

    return results_by_shots, weak_wrong_results


def plot_response_matching(
    results_by_shots: list,
    weak_name: str,
    strong_name: str,
    output_path: Path = None
):
    """Plot response matching as a function of shot count (stacked area chart).

    Args:
        results_by_shots: List of dicts with matching stats per shot count
        weak_name: Weak model name
        strong_name: Strong model name
        output_path: Optional path to save plot
    """
    # Extract data
    shot_counts = [r["num_shots"] for r in results_by_shots]
    matches_weak = [r["matches_weak"] for r in results_by_shots]
    matches_strong = [r["matches_strong"] for r in results_by_shots]
    matches_both = [r["matches_both"] for r in results_by_shots]
    matches_neither = [r["matches_neither"] for r in results_by_shots]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create stacked area chart
    # Stack order: bottom to top
    ax.fill_between(shot_counts, 0, matches_both,
                    label='Matches both', color='#2ca02c', alpha=0.7)

    ax.fill_between(shot_counts, matches_both,
                    [matches_both[i] + matches_strong[i] for i in range(len(shot_counts))],
                    label='Matches strong only', color='#ff7f0e', alpha=0.7)

    ax.fill_between(shot_counts,
                    [matches_both[i] + matches_strong[i] for i in range(len(shot_counts))],
                    [matches_both[i] + matches_strong[i] + matches_weak[i] for i in range(len(shot_counts))],
                    label='Matches weak only', color='#1f77b4', alpha=0.7)

    ax.fill_between(shot_counts,
                    [matches_both[i] + matches_strong[i] + matches_weak[i] for i in range(len(shot_counts))],
                    100,
                    label='Matches neither', color='#d62728', alpha=0.7)

    # Add percentage labels at the center of each section for middle points
    if len(shot_counts) >= 3:
        mid_idx = len(shot_counts) // 2
        x_mid = shot_counts[mid_idx]

        # Calculate y positions (middle of each band)
        y_both = matches_both[mid_idx] / 2
        y_strong = matches_both[mid_idx] + matches_strong[mid_idx] / 2
        y_weak = matches_both[mid_idx] + matches_strong[mid_idx] + matches_weak[mid_idx] / 2
        y_neither = matches_both[mid_idx] + matches_strong[mid_idx] + matches_weak[mid_idx] + matches_neither[mid_idx] / 2

        if matches_both[mid_idx] > 5:
            ax.text(x_mid, y_both, f'{matches_both[mid_idx]:.0f}%',
                   ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        if matches_strong[mid_idx] > 5:
            ax.text(x_mid, y_strong, f'{matches_strong[mid_idx]:.0f}%',
                   ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        if matches_weak[mid_idx] > 5:
            ax.text(x_mid, y_weak, f'{matches_weak[mid_idx]:.0f}%',
                   ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        if matches_neither[mid_idx] > 5:
            ax.text(x_mid, y_neither, f'{matches_neither[mid_idx]:.0f}%',
                   ha='center', va='center', fontsize=10, fontweight='bold', color='white')

    # Customize plot
    ax.set_xlabel('Number of Weak Few-Shot Examples', fontsize=13, fontweight='bold')
    ax.set_ylabel('Percentage of Test Questions (%)', fontsize=13, fontweight='bold')
    ax.set_title(f'Response Matching (All Questions): {weak_name} → {strong_name}\n' +
                 'How W2S predictions align with weak and strong models',
                 fontsize=14, fontweight='bold', pad=20)

    # Set axes
    ax.set_xlim(min(shot_counts), max(shot_counts))
    ax.set_ylim(0, 100)
    ax.set_xticks(shot_counts)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_axisbelow(True)

    # Add legend (reversed to match stacking order)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='best', fontsize=11, framealpha=0.9)

    # Add info text
    info_text = (
        f"Weak model: {weak_name}\n"
        f"Strong model: {strong_name}\n"
        f"Test size: 200 questions"
    )

    ax.text(0.02, 0.98, info_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {output_path}")
    else:
        plt.show()

    return fig, ax


def plot_response_matching_weak_wrong(
    results_by_shots: list,
    weak_name: str,
    strong_name: str,
    output_path: Path = None
):
    """Plot response matching when weak model is wrong (stacked area chart).

    Args:
        results_by_shots: List of dicts with matching stats per shot count
        weak_name: Weak model name
        strong_name: Strong model name
        output_path: Optional path to save plot
    """
    # Extract data
    shot_counts = [r["num_shots"] for r in results_by_shots]
    matches_weak = [r["matches_weak"] for r in results_by_shots]
    matches_strong = [r["matches_strong"] for r in results_by_shots]
    matches_both = [r["matches_both"] for r in results_by_shots]
    matches_neither = [r["matches_neither"] for r in results_by_shots]
    weak_wrong_counts = [r["weak_wrong_count"] for r in results_by_shots]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create stacked area chart
    # Stack order: bottom to top
    ax.fill_between(shot_counts, 0, matches_both,
                    label='Matches both (weak & strong agree)', color='#9467bd', alpha=0.7)

    ax.fill_between(shot_counts, matches_both,
                    [matches_both[i] + matches_strong[i] for i in range(len(shot_counts))],
                    label='Matches strong only (corrects to strong)', color='#2ca02c', alpha=0.7)

    ax.fill_between(shot_counts,
                    [matches_both[i] + matches_strong[i] for i in range(len(shot_counts))],
                    [matches_both[i] + matches_strong[i] + matches_weak[i] for i in range(len(shot_counts))],
                    label='Matches weak only (copies error)', color='#d62728', alpha=0.7)

    ax.fill_between(shot_counts,
                    [matches_both[i] + matches_strong[i] + matches_weak[i] for i in range(len(shot_counts))],
                    100,
                    label='Matches neither (unique answer)', color='#ff7f0e', alpha=0.7)

    # Add percentage labels at the center of each section for middle points
    if len(shot_counts) >= 3:
        mid_idx = len(shot_counts) // 2
        x_mid = shot_counts[mid_idx]

        # Calculate y positions (middle of each band)
        y_both = matches_both[mid_idx] / 2
        y_strong = matches_both[mid_idx] + matches_strong[mid_idx] / 2
        y_weak = matches_both[mid_idx] + matches_strong[mid_idx] + matches_weak[mid_idx] / 2
        y_neither = matches_both[mid_idx] + matches_strong[mid_idx] + matches_weak[mid_idx] + matches_neither[mid_idx] / 2

        if matches_both[mid_idx] > 5:
            ax.text(x_mid, y_both, f'{matches_both[mid_idx]:.0f}%',
                   ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        if matches_strong[mid_idx] > 5:
            ax.text(x_mid, y_strong, f'{matches_strong[mid_idx]:.0f}%',
                   ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        if matches_weak[mid_idx] > 5:
            ax.text(x_mid, y_weak, f'{matches_weak[mid_idx]:.0f}%',
                   ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        if matches_neither[mid_idx] > 5:
            ax.text(x_mid, y_neither, f'{matches_neither[mid_idx]:.0f}%',
                   ha='center', va='center', fontsize=10, fontweight='bold', color='white')

    # Customize plot
    ax.set_xlabel('Number of Weak Few-Shot Examples', fontsize=13, fontweight='bold')
    ax.set_ylabel('Percentage of Questions (%)', fontsize=13, fontweight='bold')
    ax.set_title(f'Response Matching When Weak Model Wrong: {weak_name} → {strong_name}\n' +
                 'Does W2S copy weak errors or correct to strong?',
                 fontsize=14, fontweight='bold', pad=20)

    # Set axes
    ax.set_xlim(min(shot_counts), max(shot_counts))
    ax.set_ylim(0, 100)
    ax.set_xticks(shot_counts)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_axisbelow(True)

    # Add legend (reversed to match stacking order)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='best', fontsize=11, framealpha=0.9)

    # Add info text with sample size
    avg_weak_wrong = sum(weak_wrong_counts) / len(weak_wrong_counts)
    info_text = (
        f"Weak model: {weak_name}\n"
        f"Strong model: {strong_name}\n"
        f"Avg questions where weak wrong: {avg_weak_wrong:.0f}"
    )

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


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Plot response matching over number of few-shot examples"
    )
    parser.add_argument(
        "--weak-model",
        type=str,
        required=True,
        help="Weak model ID (e.g., meta-llama/llama-3.1-8b-instruct)"
    )
    parser.add_argument(
        "--strong-model",
        type=str,
        required=True,
        help="Strong model ID (e.g., qwen/qwen-2.5-72b-instruct)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for plot (optional)"
    )

    args = parser.parse_args()

    # Get project root and results directory
    # Go up from plotting_and_analysis/ -> code/ -> astra-worktest/
    project_root = Path(__file__).parent.parent.parent
    weak_labels_dir = project_root / "results" / "weak_labels"

    print("=" * 80)
    print("Response Matching Analysis")
    print("=" * 80)
    print(f"\nWeak model:   {args.weak_model}")
    print(f"Strong model: {args.strong_model}")

    # Load ground truth
    print("\nLoading TruthfulQA dataset for ground truth...")
    test_set, _ = load_truthful_qa(test_size=200, random_seed=42)
    ground_truth = [q.answer for q in test_set]

    # Find W2S results file
    weak_slug = args.weak_model.replace('/', '_')
    strong_slug = args.strong_model.replace('/', '_')

    # Find the most recent file for this pair
    pattern = f"w2s_{weak_slug}_to_{strong_slug}_*.json"
    w2s_files = list(weak_labels_dir.glob(pattern))

    if not w2s_files:
        print(f"\nError: No W2S results found for this model pair")
        print(f"  Looking for: {pattern}")
        print(f"  In directory: {weak_labels_dir}")
        return

    # Use most recent
    w2s_file = max(w2s_files, key=lambda p: p.stat().st_mtime)
    print(f"\nLoading W2S results: {w2s_file.name}")
    w2s_results = load_results(w2s_file)

    # Load baselines
    print("Loading baseline results...")
    try:
        weak_baseline = get_baseline_results(weak_labels_dir, args.weak_model)
        strong_baseline = get_baseline_results(weak_labels_dir, args.strong_model)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return

    # Analyze
    print("\nAnalyzing response matching over shot counts...")
    results, weak_wrong_results = analyze_response_matching_over_shots(
        w2s_results, weak_baseline, strong_baseline, ground_truth
    )

    # Print results (all questions)
    print("\n" + "=" * 80)
    print("RESPONSE MATCHING BY SHOT COUNT (All Questions)")
    print("=" * 80)
    print(f"\n{'Shots':>6} | {'Weak':>10} | {'Strong':>10} | {'Both':>10} | {'Neither':>10}")
    print("-" * 80)
    for r in results:
        print(f"{r['num_shots']:>6} | {r['matches_weak']:>9.1f}% | "
              f"{r['matches_strong']:>9.1f}% | {r['matches_both']:>9.1f}% | "
              f"{r['matches_neither']:>9.1f}%")
    print("=" * 80)

    # Print results (weak wrong only)
    print("\n" + "=" * 80)
    print("RESPONSE MATCHING BY SHOT COUNT (Only When Weak Model Wrong)")
    print("=" * 80)
    print(f"\n{'Shots':>6} | {'Weak':>10} | {'Strong':>10} | {'Both':>10} | {'Neither':>10} | {'Count':>6}")
    print("-" * 80)
    for r in weak_wrong_results:
        print(f"{r['num_shots']:>6} | {r['matches_weak']:>9.1f}% | "
              f"{r['matches_strong']:>9.1f}% | {r['matches_both']:>9.1f}% | "
              f"{r['matches_neither']:>9.1f}% | {r['weak_wrong_count']:>6}")
    print("=" * 80)

    # Generate output paths
    if args.output:
        output_path_all = Path(args.output)
        output_path_weak_wrong = Path(args.output).parent / (Path(args.output).stem + "_weak_wrong" + Path(args.output).suffix)
    else:
        # Create plots directory structure
        plots_dir = project_root / "results" / "plots" / "response_matching"
        plots_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path_all = plots_dir / f"response_matching_all_questions_{weak_slug}_to_{strong_slug}_{timestamp}.png"
        output_path_weak_wrong = plots_dir / f"response_matching_when_weak_wrong_{weak_slug}_to_{strong_slug}_{timestamp}.png"

    # Plot 1: All questions
    print("\nGenerating plot for all questions...")
    weak_name = w2s_results["weak_model_name"]
    strong_name = w2s_results["strong_model_name"]
    plot_response_matching(results, weak_name, strong_name, output_path_all)

    # Plot 2: Only when weak is wrong
    print("\nGenerating plot for questions where weak model was wrong...")
    plot_response_matching_weak_wrong(weak_wrong_results, weak_name, strong_name, output_path_weak_wrong)

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
