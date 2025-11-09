#!/usr/bin/env python3
"""Evaluate models with gold (ground truth) few-shot labels.

This implements Experiments 1 & 2 from the PGR methodology:
- Experiment 1: Weak model with gold labels
- Experiment 2: Strong model with gold labels
"""

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

from dataset_utils import load_truthful_qa
from api_utils import OpenRouterAPI, get_batch_responses_with_few_shot
from eval_utils import score_batch


async def evaluate_with_gold_labels(
    model: str,
    num_few_shot: int = 5,
    test_size: int = 200,
    cache_dir: Path = Path("./cache")
):
    """Evaluate a model with gold (correct) few-shot labels.

    Args:
        model: Model ID (e.g., "meta-llama/llama-3.1-8b-instruct")
        num_few_shot: Number of few-shot examples to use
        test_size: Number of test questions
        cache_dir: Cache directory for API responses

    Returns:
        Dictionary with results
    """
    print("=" * 70)
    print(f"Evaluating {model} with Gold Few-Shot Labels")
    print("=" * 70)

    # 1. Load dataset
    print("\n1. Loading TruthfulQA dataset...")
    test_set, train_set = load_truthful_qa(test_size=test_size, random_seed=42)
    print(f"   Test set: {len(test_set)} questions")
    print(f"   Train set: {len(train_set)} questions")

    # 2. Initialize API
    print("\n2. Initializing OpenRouter API...")
    api = OpenRouterAPI(cache_dir=cache_dir, num_threads=50)

    # 3. Create few-shot examples with GOLD labels from train set
    print(f"\n3. Creating {num_few_shot} few-shot examples with gold labels...")
    few_shot_examples = [
        (train_set[i].question, train_set[i].answer)
        for i in range(num_few_shot)
    ]

    print(f"   Example few-shot questions (first 2):")
    for i in range(min(2, num_few_shot)):
        q_preview = train_set[i].question[:100].replace('\n', ' ')
        print(f"     {i+1}. {q_preview}... → {train_set[i].answer}")

    # 4. Evaluate model
    print(f"\n4. Evaluating {model}...")
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
        "experiment": "gold_labels",
        "dataset": {
            "name": "TruthfulQA",
            "test_size": len(test_set),
            "train_size": len(train_set),
        },
        "few_shot": {
            "num_examples": num_few_shot,
            "label_type": "gold",
            "source": "train_set",
            "indices": list(range(num_few_shot))
        },
        "results": {
            "accuracy": eval_results['accuracy'],
            "num_correct": eval_results['num_correct'],
            "num_total": eval_results['num_total'],
            "predictions": eval_results['predictions'],
        }
    }

    return results


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Evaluate models with gold few-shot labels"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/llama-3.1-8b-instruct",
        help="Model ID to evaluate (default: meta-llama/llama-3.1-8b-instruct)"
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

    # Run evaluation
    results = await evaluate_with_gold_labels(
        model=args.model,
        num_few_shot=args.num_few_shot,
        test_size=args.test_size
    )

    # Save results
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    model_slug = args.model.replace('/', '_')
    timestamp = results["timestamp"]
    num_shots = results["few_shot"]["num_examples"]
    output_file = results_dir / f"gold_labels_{model_slug}_{num_shots}shot_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Model: {results['model_name']}")
    print(f"Few-shot examples: {results['few_shot']['num_examples']} (gold labels)")
    print(f"Accuracy: {results['results']['accuracy']:.1%} ({results['results']['num_correct']}/{results['results']['num_total']})")
    print("=" * 70)
    print(f"Results saved to: {output_file}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
