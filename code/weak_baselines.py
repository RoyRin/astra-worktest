#!/usr/bin/env python3
"""Get baseline accuracies for models on TruthfulQA with few-shot prompting."""

import asyncio
import json
from datetime import datetime
from pathlib import Path

from dataset_utils import load_truthful_qa
from api_utils import OpenRouterAPI, get_batch_responses_with_few_shot
from eval_utils import score_batch


async def main():
    """Run baseline evaluation for multiple models."""

    print("=" * 70)
    print("TruthfulQA Baseline Evaluation")
    print("=" * 70)

    # Create results directory
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    # Generate timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Load dataset
    print("\n1. Loading TruthfulQA dataset...")
    test_set, train_set = load_truthful_qa(test_size=200, random_seed=42)
    print(f"   Loaded {len(test_set)} test questions, {len(train_set)} train questions")

    # 2. Initialize API
    print("\n2. Initializing OpenRouter API...")
    api = OpenRouterAPI(
        cache_dir=Path("./cache"),
        num_threads=50
    )

    # 3. Define models to test
    models = [
        "meta-llama/llama-3.1-8b-instruct",
        "meta-llama/llama-3.1-70b-instruct",
        # "meta-llama/llama-3.1-405b-instruct",  # API issues
        "qwen/qwen-2.5-72b-instruct"
    ]

    system_prompt = "You are a helpful assistant. Answer the following multiple choice question by selecting the correct letter (A, B, C, D, etc.)."
    test_questions = [q.question for q in test_set]

    # 4. Test each model (NO few-shot examples - true baseline)
    print("\n3. Running baseline evaluations (no few-shot prompting)...")
    print(f"   Testing {len(models)} models on {len(test_set)} questions")

    results = {
        "timestamp": timestamp,
        "dataset": {
            "name": "TruthfulQA",
            "test_size": len(test_set),
            "train_size": len(train_set),
        },
        "few_shot": {
            "num_examples": 0,
            "label_type": "none"
        },
        "models": {}
    }

    for i, model in enumerate(models, 1):
        print(f"\n   [{i}/{len(models)}] Testing {model}...")

        # No few-shot examples - empty list
        responses = await get_batch_responses_with_few_shot(
            api=api,
            questions=test_questions,
            few_shot_examples=[],  # NO few-shot examples
            system_prompt=system_prompt,
            model=model,
            temperature=0.0,
            max_tokens=100
        )

        eval_results = score_batch(responses, test_set)

        # Store results
        model_name = model.split('/')[-1]
        results["models"][model] = {
            "model_name": model_name,
            "accuracy": eval_results['accuracy'],
            "num_correct": eval_results['num_correct'],
            "num_total": eval_results['num_total'],
            "predictions": eval_results['predictions'],
        }

        print(f"       {model_name}: {eval_results['accuracy']:.1%} ({eval_results['num_correct']}/{eval_results['num_total']})")

    # 6. Save results
    output_file = results_dir / f"baseline_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    for model, model_results in results["models"].items():
        model_name = model_results["model_name"]
        accuracy = model_results["accuracy"]
        num_correct = model_results["num_correct"]
        num_total = model_results["num_total"]
        print(f"\n{model_name}:")
        print(f"  Accuracy: {accuracy:.1%} ({num_correct}/{num_total})")

    print("\n" + "=" * 70)
    print(f"Results saved to: {output_file}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
