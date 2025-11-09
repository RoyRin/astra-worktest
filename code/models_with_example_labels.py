#!/usr/bin/env python3
"""Evaluate models with few-shot labels (gold or weak).

This implements experiments from the PGR methodology:
- Ground truth labels: Use correct answers from train set
- Weak labels: Use predictions from a weak model
"""

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

from dataset_utils import load_truthful_qa
from api_utils import OpenRouterAPI, get_batch_responses_with_few_shot
from eval_utils import score_batch, extract_answer_letter


def load_weak_labels(weak_label_model: str, results_dir: Path) -> dict:
    """Load weak labels from baseline results.

    Args:
        weak_label_model: Model name for weak labels (e.g., "meta-llama/llama-3.1-8b-instruct")
        results_dir: Results directory path

    Returns:
        Dictionary with predictions
    """
    model_slug = weak_label_model.replace('/', '_')
    weak_label_file = results_dir / "weak_labels" / f"{model_slug}_baseline.json"

    if not weak_label_file.exists():
        raise FileNotFoundError(
            f"Weak label file not found: {weak_label_file}\n"
            f"Please run baselines.py first to generate weak labels."
        )

    with open(weak_label_file, 'r') as f:
        data = json.load(f)

    return data


async def evaluate_with_labels(
    model: str,
    label_type: str,
    weak_label_model: str = None,
    num_few_shot: int = 5,
    test_size: int = 200,
    cache_dir: Path = Path("./cache"),
    results_dir: Path = Path("./results")
):
    """Evaluate a model with few-shot labels (gold or weak).

    Args:
        model: Model ID to evaluate
        label_type: "ground_truth" or "weak_label"
        weak_label_model: Model to use for weak labels (if label_type="weak_label")
        num_few_shot: Number of few-shot examples to use
        test_size: Number of test questions
        cache_dir: Cache directory for API responses
        results_dir: Results directory

    Returns:
        Dictionary with results
    """
    print("=" * 70)
    print(f"Evaluating {model} with {label_type.replace('_', ' ').title()} Labels")
    print("=" * 70)

    # 1. Load dataset
    print("\n1. Loading TruthfulQA dataset...")
    test_set, train_set = load_truthful_qa(test_size=test_size, random_seed=42)
    print(f"   Test set: {len(test_set)} questions")
    print(f"   Train set: {len(train_set)} questions")

    # 2. Initialize API
    print("\n2. Initializing OpenRouter API...")
    api = OpenRouterAPI(cache_dir=cache_dir, num_threads=50)

    # 3. Create few-shot examples
    if label_type == "ground_truth":
        print(f"\n3. Creating {num_few_shot} few-shot examples with GOLD labels...")
        few_shot_examples = [
            (train_set[i].question, train_set[i].answer)
            for i in range(num_few_shot)
        ]
        label_source = "train_set_gold"

    elif label_type == "weak_label":
        if not weak_label_model:
            raise ValueError("weak_label_model must be specified when label_type='weak_label'")

        print(f"\n3. Loading weak labels from {weak_label_model}...")
        weak_data = load_weak_labels(weak_label_model, results_dir)

        # Get predictions from weak model baseline
        weak_predictions = weak_data["results"]["predictions"]

        # Note: The baseline was run on test_set, but we need labels for train_set
        # We need to run the weak model on train_set to get few-shot examples
        print(f"   Running {weak_label_model} on train set to get weak labels...")

        # Get weak model predictions on first num_few_shot questions from train set
        train_questions_subset = [train_set[i].question for i in range(num_few_shot)]

        weak_responses = await get_batch_responses_with_few_shot(
            api=api,
            questions=train_questions_subset,
            few_shot_examples=[],  # No few-shot for generating weak labels
            system_prompt="You are a helpful assistant. Answer the following multiple choice question by selecting the correct letter (A, B, C, D, etc.).",
            model=weak_label_model,
            temperature=0.0,
            max_tokens=100
        )

        # Extract weak labels
        weak_labels = [extract_answer_letter(resp.completion) for resp in weak_responses]

        few_shot_examples = [
            (train_set[i].question, weak_labels[i])
            for i in range(num_few_shot)
        ]
        label_source = f"weak_labels_{weak_label_model}"

        print(f"   Generated {num_few_shot} weak labels")

    else:
        raise ValueError(f"Invalid label_type: {label_type}")

    print(f"   Example few-shot questions (first 2):")
    for i in range(min(2, num_few_shot)):
        q_preview = train_set[i].question[:100].replace('\n', ' ')
        print(f"     {i+1}. {q_preview}... → {few_shot_examples[i][1]}")

    # 4. Evaluate model
    print(f"\n4. Evaluating {model} on test set...")
    system_prompt = "You are a helpful assistant. Answer the following multiple choice question by selecting the correct letter (A, B, C, D, etc.)."
    test_questions = [q.question for q in test_set]

    responses = await get_batch_responses_with_few_shot(
        api=api,
        questions=test_questions,
        few_shot_examples=few_shot_examples,
        system_prompt=system_prompt,
        model=model,
        temperature=0.0,
        max_tokens=100
    )

    # 5. Score responses
    eval_results = score_batch(responses, test_set)

    model_name = model.split('/')[-1]
    print(f"\n   ✓ {model_name}: {eval_results['accuracy']:.1%} ({eval_results['num_correct']}/{eval_results['num_total']})")

    # 6. Prepare results
    results = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "model": model,
        "model_name": model_name,
        "experiment": f"few_shot_{label_type}",
        "label_type": label_type,
        "label_source": label_source,
        "dataset": {
            "name": "TruthfulQA",
            "test_size": len(test_set),
            "train_size": len(train_set),
        },
        "few_shot": {
            "num_examples": num_few_shot,
            "label_type": label_type,
            "source": label_source,
            "indices": list(range(num_few_shot)),
            "examples": few_shot_examples  # Save the actual examples used
        },
        "results": {
            "accuracy": eval_results['accuracy'],
            "num_correct": eval_results['num_correct'],
            "num_total": eval_results['num_total'],
            "predictions": eval_results['predictions'],
        }
    }

    if label_type == "weak_label":
        results["weak_label_model"] = weak_label_model

    return results


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Evaluate models with few-shot labels (gold or weak)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/llama-3.1-8b-instruct",
        help="Model ID to evaluate (default: meta-llama/llama-3.1-8b-instruct)"
    )

    # Label type group (mutually exclusive)
    label_group = parser.add_mutually_exclusive_group(required=True)
    label_group.add_argument(
        "--ground-truth",
        action="store_true",
        help="Use ground truth (gold) labels from train set"
    )
    label_group.add_argument(
        "--weak-label",
        type=str,
        metavar="WEAK_MODEL",
        help="Use weak labels from specified model (e.g., meta-llama/llama-3.1-8b-instruct)"
    )

    parser.add_argument(
        "--num-few-shot",
        type=int,
        default=5,
        help="Number of few-shot examples (default: 5)"
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=200,
        help="Number of test questions (default: 200)"
    )

    args = parser.parse_args()

    # Determine label type
    if args.ground_truth:
        label_type = "ground_truth"
        weak_label_model = None
    else:
        label_type = "weak_label"
        weak_label_model = args.weak_label

    # Setup paths
    project_root = Path(__file__).parent.parent
    cache_dir = project_root / "cache"
    results_dir = project_root / "results"

    # Run evaluation
    results = await evaluate_with_labels(
        model=args.model,
        label_type=label_type,
        weak_label_model=weak_label_model,
        num_few_shot=args.num_few_shot,
        test_size=args.test_size,
        cache_dir=cache_dir,
        results_dir=results_dir
    )

    # Save results
    results_dir.mkdir(exist_ok=True)

    model_slug = args.model.replace('/', '_')
    timestamp = results["timestamp"]
    num_shots = results["few_shot"]["num_examples"]

    if label_type == "ground_truth":
        output_file = results_dir / f"gold_labels_{model_slug}_{num_shots}shot_{timestamp}.json"
    else:
        weak_slug = weak_label_model.replace('/', '_')
        output_file = results_dir / f"weak_labels_{weak_slug}_to_{model_slug}_{num_shots}shot_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Model: {results['model_name']}")
    print(f"Label type: {label_type.replace('_', ' ').title()}")
    if label_type == "weak_label":
        print(f"Weak label source: {weak_label_model}")
    print(f"Few-shot examples: {results['few_shot']['num_examples']}")
    print(f"Accuracy: {results['results']['accuracy']:.1%} ({results['results']['num_correct']}/{results['results']['num_total']})")
    print("=" * 70)
    print(f"Results saved to: {output_file}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
