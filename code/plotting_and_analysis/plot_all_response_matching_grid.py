#!/usr/bin/env python3
"""Plot response matching for all W2S pairs in a grid layout.

Creates a single figure with subplots organized by:
- Rows: Weak model (8b, 70b, 72b)
- Columns: Strong model (8b, 70b, 72b)
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

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

    Returns:
        Tuple of (all_results, weak_wrong_results)
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
            })
        else:
            weak_wrong_results.append({
                "num_shots": num_shots,
                "matches_weak": 0,
                "matches_strong": 0,
                "matches_both": 0,
                "matches_neither": 0,
            })

    return results_by_shots, weak_wrong_results


def analyze_w2s_correctness_conditioned_on_strong(
    w2s_results: dict,
    weak_baseline: dict,
    strong_baseline: dict,
    ground_truth: list
) -> tuple:
    """Analyze W2S correctness conditioned on strong model correctness.

    Returns:
        Tuple of (strong_wrong_results, strong_correct_results)
        - strong_wrong_results: % W2S correct when strong is wrong
        - strong_correct_results: % W2S wrong when strong is correct
    """
    weak_preds = weak_baseline["results"]["predictions"]
    strong_preds = strong_baseline["results"]["predictions"]

    strong_wrong_results = []
    strong_correct_results = []

    for shot_result in w2s_results["results_by_shot_count"]:
        num_shots = shot_result["num_few_shot"]
        w2s_preds = shot_result["predictions"]

        num_questions = len(w2s_preds)

        # Count when strong is wrong
        strong_wrong_count = 0
        w2s_correct_when_strong_wrong = 0

        # Count when strong is correct
        strong_correct_count = 0
        w2s_wrong_when_strong_correct = 0

        for i in range(num_questions):
            w2s_pred = w2s_preds[i]
            strong_pred = strong_preds[i]
            gt = ground_truth[i]

            strong_is_correct = (strong_pred == gt)
            w2s_is_correct = (w2s_pred == gt)

            if strong_is_correct:
                strong_correct_count += 1
                if not w2s_is_correct:
                    w2s_wrong_when_strong_correct += 1
            else:
                strong_wrong_count += 1
                if w2s_is_correct:
                    w2s_correct_when_strong_wrong += 1

        # Calculate percentages
        if strong_wrong_count > 0:
            pct_w2s_correct_when_strong_wrong = w2s_correct_when_strong_wrong / strong_wrong_count * 100
        else:
            pct_w2s_correct_when_strong_wrong = 0

        if strong_correct_count > 0:
            pct_w2s_wrong_when_strong_correct = w2s_wrong_when_strong_correct / strong_correct_count * 100
        else:
            pct_w2s_wrong_when_strong_correct = 0

        strong_wrong_results.append({
            "num_shots": num_shots,
            "pct_w2s_correct": pct_w2s_correct_when_strong_wrong,
        })

        strong_correct_results.append({
            "num_shots": num_shots,
            "pct_w2s_wrong": pct_w2s_wrong_when_strong_correct,
        })

    return strong_wrong_results, strong_correct_results


def plot_single_panel(ax, results, weak_name, strong_name, show_ylabel=False, show_xlabel=False):
    """Plot a single panel of the grid."""
    # Extract data
    shot_counts = [r["num_shots"] for r in results]
    matches_weak = [r["matches_weak"] for r in results]
    matches_strong = [r["matches_strong"] for r in results]
    matches_both = [r["matches_both"] for r in results]
    matches_neither = [r["matches_neither"] for r in results]

    # Create stacked area chart
    ax.fill_between(shot_counts, 0, matches_both,
                    label='Both', color='#2ca02c', alpha=0.7)

    ax.fill_between(shot_counts, matches_both,
                    [matches_both[i] + matches_strong[i] for i in range(len(shot_counts))],
                    label='Strong', color='#ff7f0e', alpha=0.7)

    ax.fill_between(shot_counts,
                    [matches_both[i] + matches_strong[i] for i in range(len(shot_counts))],
                    [matches_both[i] + matches_strong[i] + matches_weak[i] for i in range(len(shot_counts))],
                    label='Weak', color='#1f77b4', alpha=0.7)

    ax.fill_between(shot_counts,
                    [matches_both[i] + matches_strong[i] + matches_weak[i] for i in range(len(shot_counts))],
                    100,
                    label='Neither', color='#d62728', alpha=0.7)

    # Customize
    ax.set_xlim(min(shot_counts), max(shot_counts))
    ax.set_ylim(0, 100)
    ax.set_xticks(shot_counts)
    ax.tick_params(labelsize=8)

    if show_xlabel:
        ax.set_xlabel('# Few-Shot Examples', fontsize=9)
    if show_ylabel:
        ax.set_ylabel('Percentage (%)', fontsize=9)

    # Grid
    ax.grid(True, alpha=0.2, linestyle='--', axis='y')
    ax.set_axisbelow(True)

    # Title showing weak → strong
    ax.set_title(f'{weak_name} → {strong_name}', fontsize=9, fontweight='bold')


def plot_conditional_correctness_panel(ax, strong_wrong_data, strong_correct_data, weak_name, strong_name,
                                       show_ylabel=False, show_xlabel=False):
    """Plot W2S correctness conditioned on strong model performance.

    Two lines:
    - W2S correct when strong is wrong (higher is better - shows W2S fixing strong's errors)
    - W2S wrong when strong is correct (lower is better - shows W2S not breaking strong's successes)
    """
    shot_counts_wrong = [r["num_shots"] for r in strong_wrong_data]
    pct_w2s_correct_when_strong_wrong = [r["pct_w2s_correct"] for r in strong_wrong_data]

    shot_counts_correct = [r["num_shots"] for r in strong_correct_data]
    pct_w2s_wrong_when_strong_correct = [r["pct_w2s_wrong"] for r in strong_correct_data]

    # Plot both lines
    ax.plot(shot_counts_wrong, pct_w2s_correct_when_strong_wrong,
            marker='o', color='#2ca02c', linewidth=2, markersize=6,
            label='W2S correct | Strong wrong', alpha=0.8)

    ax.plot(shot_counts_correct, pct_w2s_wrong_when_strong_correct,
            marker='s', color='#d62728', linewidth=2, markersize=6,
            label='W2S wrong | Strong correct', alpha=0.8)

    # Customize
    ax.set_xlim(min(shot_counts_wrong), max(shot_counts_wrong))
    ax.set_ylim(0, 100)
    ax.set_xticks(shot_counts_wrong)
    ax.tick_params(labelsize=8)

    if show_xlabel:
        ax.set_xlabel('# Few-Shot Examples', fontsize=9)
    if show_ylabel:
        ax.set_ylabel('Percentage (%)', fontsize=9)

    # Grid
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.set_axisbelow(True)

    # Title showing weak → strong
    ax.set_title(f'{weak_name} → {strong_name}', fontsize=9, fontweight='bold')


def main():
    """Main function."""
    # Get project root and results directory
    project_root = Path(__file__).parent.parent.parent
    weak_labels_dir = project_root / "results" / "weak_labels"

    print("=" * 80)
    print("Response Matching Grid Plot")
    print("=" * 80)

    # Load ground truth
    print("\nLoading TruthfulQA dataset for ground truth...")
    test_set, _ = load_truthful_qa(test_size=200, random_seed=42)
    ground_truth = [q.answer for q in test_set]

    # Define models
    models = [
        "meta-llama/llama-3.1-8b-instruct",
        "meta-llama/llama-3.1-70b-instruct",
        "qwen/qwen-2.5-72b-instruct"
    ]

    model_short_names = {
        "meta-llama/llama-3.1-8b-instruct": "Llama-8B",
        "meta-llama/llama-3.1-70b-instruct": "Llama-70B",
        "qwen/qwen-2.5-72b-instruct": "Qwen-72B"
    }

    # Load baselines once
    print("\nLoading baselines...")
    baselines = {}
    for model in models:
        try:
            baselines[model] = get_baseline_results(weak_labels_dir, model)
        except FileNotFoundError as e:
            print(f"  Error loading baseline for {model}: {e}")
            return

    # Analyze all pairs once
    print("\nProcessing model pairs...")
    all_results_dict = {}
    weak_wrong_results_dict = {}
    conditional_correctness_dict = {}

    for i, weak_model in enumerate(models):
        for j, strong_model in enumerate(models):
            weak_short = model_short_names[weak_model]
            strong_short = model_short_names[strong_model]

            print(f"  [{i},{j}] {weak_short} → {strong_short}")

            # Find W2S results file
            weak_slug = weak_model.replace('/', '_')
            strong_slug = strong_model.replace('/', '_')
            pattern = f"w2s_{weak_slug}_to_{strong_slug}_*.json"
            w2s_files = list(weak_labels_dir.glob(pattern))

            if not w2s_files:
                print(f"    Warning: No results found")
                all_results_dict[(i, j)] = None
                weak_wrong_results_dict[(i, j)] = None
                conditional_correctness_dict[(i, j)] = None
                continue

            # Use most recent
            w2s_file = max(w2s_files, key=lambda p: p.stat().st_mtime)
            w2s_results = load_results(w2s_file)

            # Analyze response matching
            results, weak_wrong = analyze_response_matching_over_shots(
                w2s_results,
                baselines[weak_model],
                baselines[strong_model],
                ground_truth
            )

            # Analyze conditional correctness
            strong_wrong, strong_correct = analyze_w2s_correctness_conditioned_on_strong(
                w2s_results,
                baselines[weak_model],
                baselines[strong_model],
                ground_truth
            )

            all_results_dict[(i, j)] = (results, weak_short, strong_short)
            weak_wrong_results_dict[(i, j)] = (weak_wrong, weak_short, strong_short)
            conditional_correctness_dict[(i, j)] = (strong_wrong, strong_correct, weak_short, strong_short)

    # Create Figure 1: All questions
    print("\n" + "=" * 80)
    print("Creating grid plot 1: All Questions")
    print("=" * 80)

    fig1, axes1 = plt.subplots(3, 3, figsize=(16, 12))

    for (i, j), data in all_results_dict.items():
        if data is None:
            axes1[i, j].text(0.5, 0.5, 'No data', ha='center', va='center',
                           transform=axes1[i, j].transAxes)
            axes1[i, j].set_xticks([])
            axes1[i, j].set_yticks([])
        else:
            results, weak_short, strong_short = data
            show_ylabel = (j == 0)
            show_xlabel = (i == 2)
            plot_single_panel(axes1[i, j], results, weak_short, strong_short,
                            show_ylabel, show_xlabel)

    # Add row labels (weak models) on the left
    for i, weak_model in enumerate(models):
        weak_short = model_short_names[weak_model]
        fig1.text(0.02, 0.83 - i * 0.31, f'Weak:\n{weak_short}',
                va='center', ha='center', fontsize=11, fontweight='bold',
                rotation=90)

    # Overall title
    fig1.suptitle('Response Matching (All Questions): W2S Alignment with Weak and Strong Models\n' +
                 'Stacked Area Charts Across All Model Pairs',
                 fontsize=14, fontweight='bold', y=0.99)

    # Add legend at the bottom
    handles, labels = axes1[0, 0].get_legend_handles_labels()
    fig1.legend(handles[::-1], labels[::-1],
              loc='lower center', ncol=4, fontsize=10,
              bbox_to_anchor=(0.5, -0.01), framealpha=0.9)

    # Adjust layout
    fig1.tight_layout(rect=[0.04, 0.02, 1, 0.95])

    # Save
    plots_dir = project_root / "results" / "plots" / "response_matching"
    plots_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path1 = plots_dir / f"response_matching_all_questions_grid_{timestamp}.png"

    fig1.savefig(output_path1, dpi=300, bbox_inches='tight')
    print(f"\n{'=' * 80}")
    print(f"Grid plot (all questions) saved to: {output_path1}")
    print(f"{'=' * 80}\n")
    plt.close(fig1)

    # Create Figure 2: When weak wrong
    print("\n" + "=" * 80)
    print("Creating grid plot 2: When Weak Model Wrong")
    print("=" * 80)

    fig2, axes2 = plt.subplots(3, 3, figsize=(16, 12))

    for (i, j), data in weak_wrong_results_dict.items():
        if data is None:
            axes2[i, j].text(0.5, 0.5, 'No data', ha='center', va='center',
                           transform=axes2[i, j].transAxes)
            axes2[i, j].set_xticks([])
            axes2[i, j].set_yticks([])
        else:
            results, weak_short, strong_short = data
            show_ylabel = (j == 0)
            show_xlabel = (i == 2)
            plot_single_panel(axes2[i, j], results, weak_short, strong_short,
                            show_ylabel, show_xlabel)

    # Add row labels (weak models) on the left
    for i, weak_model in enumerate(models):
        weak_short = model_short_names[weak_model]
        fig2.text(0.02, 0.83 - i * 0.31, f'Weak:\n{weak_short}',
                va='center', ha='center', fontsize=11, fontweight='bold',
                rotation=90)

    # Overall title
    fig2.suptitle('Response Matching (When Weak Wrong): W2S Alignment with Weak and Strong Models\n' +
                 'Stacked Area Charts Across All Model Pairs',
                 fontsize=14, fontweight='bold', y=0.99)

    # Add legend at the bottom
    handles, labels = axes2[0, 0].get_legend_handles_labels()
    fig2.legend(handles[::-1], labels[::-1],
              loc='lower center', ncol=4, fontsize=10,
              bbox_to_anchor=(0.5, -0.01), framealpha=0.9)

    # Adjust layout
    fig2.tight_layout(rect=[0.04, 0.02, 1, 0.95])

    # Save
    output_path2 = plots_dir / f"response_matching_when_weak_wrong_grid_{timestamp}.png"

    fig2.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"\n{'=' * 80}")
    print(f"Grid plot (when weak wrong) saved to: {output_path2}")
    print(f"{'=' * 80}\n")
    plt.close(fig2)

    # Create Figure 3: W2S correctness conditioned on strong model performance
    print("\n" + "=" * 80)
    print("Creating grid plot 3: W2S Correctness Conditioned on Strong Performance")
    print("=" * 80)

    fig3, axes3 = plt.subplots(3, 3, figsize=(16, 12))

    for (i, j), data in conditional_correctness_dict.items():
        if data is None:
            axes3[i, j].text(0.5, 0.5, 'No data', ha='center', va='center',
                           transform=axes3[i, j].transAxes)
            axes3[i, j].set_xticks([])
            axes3[i, j].set_yticks([])
        else:
            strong_wrong, strong_correct, weak_short, strong_short = data
            show_ylabel = (j == 0)
            show_xlabel = (i == 2)
            plot_conditional_correctness_panel(axes3[i, j], strong_wrong, strong_correct,
                                              weak_short, strong_short,
                                              show_ylabel, show_xlabel)

    # Add row labels (weak models) on the left
    for i, weak_model in enumerate(models):
        weak_short = model_short_names[weak_model]
        fig3.text(0.02, 0.83 - i * 0.31, f'Weak:\n{weak_short}',
                va='center', ha='center', fontsize=11, fontweight='bold',
                rotation=90)

    # Overall title
    fig3.suptitle('W2S Correctness Conditioned on Strong Model Performance\n' +
                 'Green (↑ better): W2S fixes strong errors | Red (↓ better): W2S breaks strong successes',
                 fontsize=14, fontweight='bold', y=0.99)

    # Add legend at the bottom
    handles, labels = axes3[0, 0].get_legend_handles_labels()
    fig3.legend(handles, labels,
              loc='lower center', ncol=2, fontsize=10,
              bbox_to_anchor=(0.5, -0.01), framealpha=0.9)

    # Adjust layout
    fig3.tight_layout(rect=[0.04, 0.02, 1, 0.95])

    # Save
    output_path3 = plots_dir / f"w2s_conditional_correctness_grid_{timestamp}.png"

    fig3.savefig(output_path3, dpi=300, bbox_inches='tight')
    print(f"\n{'=' * 80}")
    print(f"Grid plot (conditional correctness) saved to: {output_path3}")
    print(f"{'=' * 80}\n")
    plt.close(fig3)


if __name__ == "__main__":
    main()
