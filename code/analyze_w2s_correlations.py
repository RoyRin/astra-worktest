#!/usr/bin/env python3
"""Analyze correlations between weak, strong, and weak-to-strong model predictions.

This script examines:
1. Agreement patterns: when do models agree/disagree?
2. Error patterns: how do models fail differently?
3. Exact response matching: are W2S responses identical to weak or strong?
4. Conditional accuracies: performance based on other models' correctness
"""

import json
import argparse
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dataset_utils import load_truthful_qa


def load_results(results_path: Path) -> dict:
    """Load results from JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)


def get_latest_w2s_results(weak_labels_dir: Path) -> List[Path]:
    """Get the most recent W2S sweep results files."""
    result_files = list(weak_labels_dir.glob("w2s_*.json"))
    if not result_files:
        raise FileNotFoundError(f"No W2S results found in {weak_labels_dir}")

    # Group by timestamp (last part before .json)
    by_timestamp = defaultdict(list)
    for f in result_files:
        # Extract timestamp from filename: w2s_..._to_..._{timestamp}.json
        parts = f.stem.split('_')
        timestamp = parts[-1]
        by_timestamp[timestamp].append(f)

    # Get most recent timestamp
    latest_timestamp = max(by_timestamp.keys())
    return by_timestamp[latest_timestamp]


def get_baseline_results(weak_labels_dir: Path, model: str) -> dict:
    """Get baseline results for a model."""
    model_slug = model.replace('/', '_')
    baseline_file = weak_labels_dir / f"{model_slug}_baseline.json"

    if not baseline_file.exists():
        raise FileNotFoundError(f"Baseline not found: {baseline_file}")

    return load_results(baseline_file)


def analyze_pair(
    weak_model: str,
    strong_model: str,
    w2s_results: dict,
    weak_baseline: dict,
    strong_baseline: dict,
    ground_truth: list,
    num_shots: int = 0
) -> dict:
    """Analyze a single weak-to-strong model pair at a specific shot count.

    Args:
        weak_model: Weak model ID
        strong_model: Strong model ID
        w2s_results: W2S results data
        weak_baseline: Weak model baseline results
        strong_baseline: Strong model baseline results
        ground_truth: List of ground truth answers
        num_shots: Number of shots to analyze (default: 0 for zero-shot)

    Returns:
        Dictionary with analysis results
    """
    # Get predictions for the specified shot count
    w2s_shot_results = None
    for result in w2s_results["results_by_shot_count"]:
        if result["num_few_shot"] == num_shots:
            w2s_shot_results = result
            break

    if w2s_shot_results is None:
        raise ValueError(f"No results found for {num_shots} shots")

    # Predictions are just letter strings
    w2s_preds = w2s_shot_results["predictions"]
    weak_preds = weak_baseline["results"]["predictions"]
    strong_preds = strong_baseline["results"]["predictions"]

    num_questions = len(w2s_preds)

    # Create correctness arrays
    weak_correct = [weak_preds[i] == ground_truth[i] for i in range(num_questions)]
    strong_correct = [strong_preds[i] == ground_truth[i] for i in range(num_questions)]
    w2s_correct = [w2s_preds[i] == ground_truth[i] for i in range(num_questions)]

    # Initialize counters
    analysis = {
        "num_questions": num_questions,
        "num_shots": num_shots,

        # Accuracies
        "weak_accuracy": sum(weak_correct) / num_questions,
        "strong_accuracy": sum(strong_correct) / num_questions,
        "w2s_accuracy": sum(w2s_correct) / num_questions,

        # Agreement patterns
        "all_correct": 0,
        "all_wrong": 0,
        "only_weak_correct": 0,
        "only_strong_correct": 0,
        "only_w2s_correct": 0,
        "weak_strong_correct_w2s_wrong": 0,
        "weak_w2s_correct_strong_wrong": 0,
        "strong_w2s_correct_weak_wrong": 0,

        # Error analysis: W2S gets wrong when others get right
        "w2s_wrong_weak_right": 0,
        "w2s_wrong_strong_right": 0,
        "w2s_wrong_both_right": 0,

        # Success analysis: W2S gets right when others get wrong
        "w2s_right_weak_wrong": 0,
        "w2s_right_strong_wrong": 0,
        "w2s_right_both_wrong": 0,

        # Exact response matching
        "w2s_matches_weak": 0,
        "w2s_matches_strong": 0,
        "w2s_matches_both": 0,
        "w2s_matches_neither": 0,

        # Response matching when wrong
        "w2s_wrong_matches_weak_wrong": 0,
        "w2s_wrong_matches_strong_wrong": 0,
        "w2s_wrong_unique": 0,

        # Conditional accuracies
        "w2s_acc_given_weak_correct": None,
        "w2s_acc_given_weak_wrong": None,
        "w2s_acc_given_strong_correct": None,
        "w2s_acc_given_strong_wrong": None,
        "w2s_acc_given_both_correct": None,
        "w2s_acc_given_both_wrong": None,
    }

    # Track for conditional accuracies
    w2s_correct_when_weak_correct = 0
    weak_correct_count = 0
    w2s_correct_when_weak_wrong = 0
    weak_wrong_count = 0
    w2s_correct_when_strong_correct = 0
    strong_correct_count = 0
    w2s_correct_when_strong_wrong = 0
    strong_wrong_count = 0
    w2s_correct_when_both_correct = 0
    both_correct_count = 0
    w2s_correct_when_both_wrong = 0
    both_wrong_count = 0

    # Analyze each question
    for i in range(num_questions):
        w_correct = weak_correct[i]
        s_correct = strong_correct[i]
        w2s_cor = w2s_correct[i]

        w_pred = weak_preds[i]
        s_pred = strong_preds[i]
        w2s_pred = w2s_preds[i]

        # Agreement patterns
        if w_correct and s_correct and w2s_cor:
            analysis["all_correct"] += 1
        elif not w_correct and not s_correct and not w2s_cor:
            analysis["all_wrong"] += 1
        elif w_correct and not s_correct and not w2s_cor:
            analysis["only_weak_correct"] += 1
        elif not w_correct and s_correct and not w2s_cor:
            analysis["only_strong_correct"] += 1
        elif not w_correct and not s_correct and w2s_cor:
            analysis["only_w2s_correct"] += 1
        elif w_correct and s_correct and not w2s_cor:
            analysis["weak_strong_correct_w2s_wrong"] += 1
        elif w_correct and not s_correct and w2s_cor:
            analysis["weak_w2s_correct_strong_wrong"] += 1
        elif not w_correct and s_correct and w2s_cor:
            analysis["strong_w2s_correct_weak_wrong"] += 1

        # Error analysis
        if not w2s_cor:
            if w_correct:
                analysis["w2s_wrong_weak_right"] += 1
            if s_correct:
                analysis["w2s_wrong_strong_right"] += 1
            if w_correct and s_correct:
                analysis["w2s_wrong_both_right"] += 1

        # Success analysis
        if w2s_cor:
            if not w_correct:
                analysis["w2s_right_weak_wrong"] += 1
            if not s_correct:
                analysis["w2s_right_strong_wrong"] += 1
            if not w_correct and not s_correct:
                analysis["w2s_right_both_wrong"] += 1

        # Exact response matching
        matches_weak = (w2s_pred == w_pred)
        matches_strong = (w2s_pred == s_pred)

        if matches_weak and matches_strong:
            analysis["w2s_matches_both"] += 1
        elif matches_weak:
            analysis["w2s_matches_weak"] += 1
        elif matches_strong:
            analysis["w2s_matches_strong"] += 1
        else:
            analysis["w2s_matches_neither"] += 1

        # Response matching when W2S is wrong
        if not w2s_cor:
            if matches_weak and not w_correct:
                analysis["w2s_wrong_matches_weak_wrong"] += 1
            if matches_strong and not s_correct:
                analysis["w2s_wrong_matches_strong_wrong"] += 1
            if not matches_weak and not matches_strong:
                analysis["w2s_wrong_unique"] += 1

        # Conditional accuracy tracking
        if w_correct:
            weak_correct_count += 1
            if w2s_cor:
                w2s_correct_when_weak_correct += 1
        else:
            weak_wrong_count += 1
            if w2s_cor:
                w2s_correct_when_weak_wrong += 1

        if s_correct:
            strong_correct_count += 1
            if w2s_cor:
                w2s_correct_when_strong_correct += 1
        else:
            strong_wrong_count += 1
            if w2s_cor:
                w2s_correct_when_strong_wrong += 1

        if w_correct and s_correct:
            both_correct_count += 1
            if w2s_cor:
                w2s_correct_when_both_correct += 1
        elif not w_correct and not s_correct:
            both_wrong_count += 1
            if w2s_cor:
                w2s_correct_when_both_wrong += 1

    # Calculate conditional accuracies
    if weak_correct_count > 0:
        analysis["w2s_acc_given_weak_correct"] = w2s_correct_when_weak_correct / weak_correct_count
    if weak_wrong_count > 0:
        analysis["w2s_acc_given_weak_wrong"] = w2s_correct_when_weak_wrong / weak_wrong_count
    if strong_correct_count > 0:
        analysis["w2s_acc_given_strong_correct"] = w2s_correct_when_strong_correct / strong_correct_count
    if strong_wrong_count > 0:
        analysis["w2s_acc_given_strong_wrong"] = w2s_correct_when_strong_wrong / strong_wrong_count
    if both_correct_count > 0:
        analysis["w2s_acc_given_both_correct"] = w2s_correct_when_both_correct / both_correct_count
    if both_wrong_count > 0:
        analysis["w2s_acc_given_both_wrong"] = w2s_correct_when_both_wrong / both_wrong_count

    return analysis


def print_analysis(analysis: dict, weak_name: str, strong_name: str):
    """Print analysis results in a readable format."""

    print("\n" + "=" * 80)
    print(f"Analysis: {weak_name} → {strong_name} ({analysis['num_shots']} shots)")
    print("=" * 80)

    # Accuracies
    print(f"\nAccuracies:")
    print(f"  Weak:   {analysis['weak_accuracy']:.1%}")
    print(f"  Strong: {analysis['strong_accuracy']:.1%}")
    print(f"  W2S:    {analysis['w2s_accuracy']:.1%}")

    # Agreement patterns
    print(f"\nAgreement Patterns:")
    print(f"  All correct:                    {analysis['all_correct']:4d} ({analysis['all_correct']/analysis['num_questions']:.1%})")
    print(f"  All wrong:                      {analysis['all_wrong']:4d} ({analysis['all_wrong']/analysis['num_questions']:.1%})")
    print(f"  Weak & Strong correct, W2S wrong: {analysis['weak_strong_correct_w2s_wrong']:4d} ({analysis['weak_strong_correct_w2s_wrong']/analysis['num_questions']:.1%})")
    print(f"  Only W2S correct:               {analysis['only_w2s_correct']:4d} ({analysis['only_w2s_correct']/analysis['num_questions']:.1%})")

    # Error patterns
    print(f"\nW2S Error Patterns:")
    print(f"  W2S wrong when weak right:      {analysis['w2s_wrong_weak_right']:4d} ({analysis['w2s_wrong_weak_right']/analysis['num_questions']:.1%})")
    print(f"  W2S wrong when strong right:    {analysis['w2s_wrong_strong_right']:4d} ({analysis['w2s_wrong_strong_right']/analysis['num_questions']:.1%})")
    print(f"  W2S wrong when both right:      {analysis['w2s_wrong_both_right']:4d} ({analysis['w2s_wrong_both_right']/analysis['num_questions']:.1%})")

    # Success patterns
    print(f"\nW2S Success Patterns:")
    print(f"  W2S right when weak wrong:      {analysis['w2s_right_weak_wrong']:4d} ({analysis['w2s_right_weak_wrong']/analysis['num_questions']:.1%})")
    print(f"  W2S right when strong wrong:    {analysis['w2s_right_strong_wrong']:4d} ({analysis['w2s_right_strong_wrong']/analysis['num_questions']:.1%})")
    print(f"  W2S right when both wrong:      {analysis['w2s_right_both_wrong']:4d} ({analysis['w2s_right_both_wrong']/analysis['num_questions']:.1%})")

    # Response matching
    print(f"\nResponse Matching:")
    print(f"  W2S matches weak:               {analysis['w2s_matches_weak']:4d} ({analysis['w2s_matches_weak']/analysis['num_questions']:.1%})")
    print(f"  W2S matches strong:             {analysis['w2s_matches_strong']:4d} ({analysis['w2s_matches_strong']/analysis['num_questions']:.1%})")
    print(f"  W2S matches both:               {analysis['w2s_matches_both']:4d} ({analysis['w2s_matches_both']/analysis['num_questions']:.1%})")
    print(f"  W2S matches neither:            {analysis['w2s_matches_neither']:4d} ({analysis['w2s_matches_neither']/analysis['num_questions']:.1%})")

    # Response matching when wrong
    print(f"\nW2S Wrong Response Matching:")
    print(f"  Matches weak (both wrong):      {analysis['w2s_wrong_matches_weak_wrong']:4d}")
    print(f"  Matches strong (both wrong):    {analysis['w2s_wrong_matches_strong_wrong']:4d}")
    print(f"  Unique error:                   {analysis['w2s_wrong_unique']:4d}")

    # Conditional accuracies
    print(f"\nConditional W2S Accuracies:")
    if analysis['w2s_acc_given_weak_correct'] is not None:
        print(f"  Given weak correct:             {analysis['w2s_acc_given_weak_correct']:.1%}")
    if analysis['w2s_acc_given_weak_wrong'] is not None:
        print(f"  Given weak wrong:               {analysis['w2s_acc_given_weak_wrong']:.1%}")
    if analysis['w2s_acc_given_strong_correct'] is not None:
        print(f"  Given strong correct:           {analysis['w2s_acc_given_strong_correct']:.1%}")
    if analysis['w2s_acc_given_strong_wrong'] is not None:
        print(f"  Given strong wrong:             {analysis['w2s_acc_given_strong_wrong']:.1%}")
    if analysis['w2s_acc_given_both_correct'] is not None:
        print(f"  Given both correct:             {analysis['w2s_acc_given_both_correct']:.1%}")
    if analysis['w2s_acc_given_both_wrong'] is not None:
        print(f"  Given both wrong:               {analysis['w2s_acc_given_both_wrong']:.1%}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Analyze correlations between weak, strong, and W2S predictions"
    )
    parser.add_argument(
        "--num-shots",
        type=int,
        default=0,
        help="Number of shots to analyze (default: 0)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file for results (optional)"
    )

    args = parser.parse_args()

    # Get project root and results directory
    project_root = Path(__file__).parent.parent
    weak_labels_dir = project_root / "results" / "weak_labels"

    print("=" * 80)
    print("W2S Correlation Analysis")
    print("=" * 80)
    print(f"\nAnalyzing {args.num_shots}-shot results...")

    # Load ground truth
    print("\nLoading TruthfulQA dataset for ground truth...")
    test_set, _ = load_truthful_qa(test_size=200, random_seed=42)
    ground_truth = [q.answer for q in test_set]
    print(f"Loaded {len(ground_truth)} ground truth answers")

    # Get W2S results
    print(f"\nLoading W2S results from: {weak_labels_dir}")
    w2s_files = get_latest_w2s_results(weak_labels_dir)
    print(f"Found {len(w2s_files)} W2S result files")

    all_analyses = []

    for w2s_file in sorted(w2s_files):
        print(f"\nProcessing: {w2s_file.name}")

        w2s_results = load_results(w2s_file)
        weak_model = w2s_results["weak_model"]
        strong_model = w2s_results["strong_model"]
        weak_name = w2s_results["weak_model_name"]
        strong_name = w2s_results["strong_model_name"]

        # Load baselines
        try:
            weak_baseline = get_baseline_results(weak_labels_dir, weak_model)
            strong_baseline = get_baseline_results(weak_labels_dir, strong_model)
        except FileNotFoundError as e:
            print(f"  Skipping: {e}")
            continue

        # Analyze
        analysis = analyze_pair(
            weak_model, strong_model, w2s_results,
            weak_baseline, strong_baseline,
            ground_truth,
            num_shots=args.num_shots
        )

        analysis["weak_model"] = weak_model
        analysis["strong_model"] = strong_model
        analysis["weak_model_name"] = weak_name
        analysis["strong_model_name"] = strong_name

        all_analyses.append(analysis)

        print_analysis(analysis, weak_name, strong_name)

    # Save to CSV if requested
    if args.output:
        output_path = Path(args.output)
        df = pd.DataFrame(all_analyses)
        df.to_csv(output_path, index=False)
        print(f"\n\nResults saved to: {output_path}")

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
