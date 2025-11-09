#!/usr/bin/env python3
"""Sweep over number of gold few-shot examples for multiple models.

Evaluates models with varying numbers of gold few-shot examples:
0, 5, 10, 25, 50, 100
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset_utils import load_truthful_qa
from api_utils import OpenRouterAPI, get_batch_responses_with_few_shot
from eval_utils import score_batch


async def evaluate_model_with_n_shots(
    api: OpenRouterAPI,
    model: str,
    num_few_shot: int,
    test_set: list,
    train_set: list,
    system_prompt: str
) -> dict:
    """Evaluate a model with N few-shot examples.

    Args:
        api: OpenRouterAPI instance
        model: Model ID
        num_few_shot: Number of few-shot examples (0 for zero-shot)
        test_set: Test questions
        train_set: Train questions (for few-shot examples)
        system_prompt: System prompt

    Returns:
        Dictionary with results
    """
    model_name = model.split('/')[-1]
    print(f"   [{model_name}] {num_few_shot} shots...", end=" ", flush=True)

    # Create few-shot examples
    if num_few_shot == 0:
        few_shot_examples = []
    else:
        few_shot_examples = [
            (train_set[i].question, train_set[i].answer)
            for i in range(num_few_shot)
        ]

    # Evaluate
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

    # Score
    eval_results = score_batch(responses, test_set)

    print(f"{eval_results['accuracy']:.1%} ({eval_results['num_correct']}/{eval_results['num_total']})")

    return {
        "num_few_shot": num_few_shot,
        "accuracy": eval_results['accuracy'],
        "num_correct": eval_results['num_correct'],
        "num_total": eval_results['num_total'],
    }


async def main():
    """Main function."""

    print("=" * 70)
    print("Gold Few-Shot Sweep: Varying Number of Examples")
    print("=" * 70)

    # Configuration
    num_shots_list = [0, 5, 10, 25, 50, 100]
    models = [
        "meta-llama/llama-3.1-8b-instruct",
        "meta-llama/llama-3.1-70b-instruct",
        "qwen/qwen-2.5-72b-instruct"
    ]
    test_size = 200

    # 1. Load dataset
    print("\n1. Loading TruthfulQA dataset...")
    test_set, train_set = load_truthful_qa(test_size=test_size, random_seed=42)
    print(f"   Test set: {len(test_set)} questions")
    print(f"   Train set: {len(train_set)} questions")

    # Check we have enough train examples
    max_shots = max(num_shots_list)
    if len(train_set) < max_shots:
        print(f"\n   WARNING: Only {len(train_set)} train examples available, but need {max_shots}")
        print(f"   Limiting to {len(train_set)} shots maximum")
        num_shots_list = [n for n in num_shots_list if n <= len(train_set)]

    # 2. Initialize API
    print("\n2. Initializing OpenRouter API...")
    # Cache in project root, not in code/
    cache_dir = Path(__file__).parent.parent.parent / "cache"
    api = OpenRouterAPI(cache_dir=cache_dir, num_threads=50)

    system_prompt = "You are a helpful assistant. Answer the following multiple choice question by selecting the correct letter (A, B, C, D, etc.)."

    # 3. Run sweep
    print(f"\n3. Running sweep over {len(num_shots_list)} shot counts for {len(models)} models...")
    print(f"   Shot counts: {num_shots_list}\n")

    all_results = {}

    for model in models:
        model_name = model.split('/')[-1]
        print(f"\n   Model: {model_name}")

        model_results = []

        for num_shots in num_shots_list:
            result = await evaluate_model_with_n_shots(
                api=api,
                model=model,
                num_few_shot=num_shots,
                test_set=test_set,
                train_set=train_set,
                system_prompt=system_prompt
            )
            model_results.append(result)

        all_results[model] = {
            "model_name": model_name,
            "results": model_results
        }

    # 4. Save results
    # Go up to code/, then up to project root, then to results/
    results_dir = Path(__file__).parent.parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = results_dir / f"gold_sweep_{timestamp}.json"

    output_data = {
        "timestamp": timestamp,
        "experiment": "gold_labels_sweep",
        "num_shots_list": num_shots_list,
        "dataset": {
            "name": "TruthfulQA",
            "test_size": len(test_set),
            "train_size": len(train_set),
        },
        "models": all_results
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    # 5. Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    for model, data in all_results.items():
        model_name = data["model_name"]
        print(f"\n{model_name}:")
        for result in data["results"]:
            num_shots = result["num_few_shot"]
            accuracy = result["accuracy"]
            print(f"  {num_shots:3d} shots: {accuracy:.1%}")

    print("\n" + "=" * 70)
    print(f"Results saved to: {output_file}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
