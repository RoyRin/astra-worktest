#!/usr/bin/env python3
"""Sweep over number of weak few-shot examples for multiple model pairs.

Evaluates strong models with varying numbers of weak few-shot examples:
0, 5, 10, 25, 50, 100

Tests weak -> strong model pairs:
- llama-8b -> llama-8b
- llama-8b -> llama-70b
- llama-8b -> qwen-72b
- llama-70b -> llama-70b
- llama-70b -> qwen-72b
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
from eval_utils import score_batch, extract_answer_letter


async def evaluate_with_weak_labels(
    api: OpenRouterAPI,
    weak_model: str,
    strong_model: str,
    num_few_shot: int,
    test_set: list,
    train_set: list,
    system_prompt: str
) -> dict:
    """Evaluate a strong model with weak few-shot labels.

    Args:
        api: OpenRouterAPI instance
        weak_model: Model ID for generating weak labels
        strong_model: Model ID to evaluate
        num_few_shot: Number of few-shot examples (0 for zero-shot)
        test_set: Test questions
        train_set: Train questions (for few-shot examples)
        system_prompt: System prompt

    Returns:
        Dictionary with results
    """
    weak_name = weak_model.split('/')[-1]
    strong_name = strong_model.split('/')[-1]
    print(f"   [{weak_name} → {strong_name}] {num_few_shot} shots...", end=" ", flush=True)

    # Create few-shot examples
    if num_few_shot == 0:
        few_shot_examples = []
    else:
        # Get weak model predictions on first num_few_shot questions from train set
        train_questions_subset = [train_set[i].question for i in range(num_few_shot)]

        weak_responses = await get_batch_responses_with_few_shot(
            api=api,
            questions=train_questions_subset,
            few_shot_examples=[],  # No few-shot for generating weak labels
            system_prompt=system_prompt,
            model=weak_model,
            temperature=0.0,
            max_tokens=100
        )

        # Extract weak labels
        weak_labels = [extract_answer_letter(resp.completion) for resp in weak_responses]

        few_shot_examples = [
            (train_set[i].question, weak_labels[i])
            for i in range(num_few_shot)
        ]

    # Evaluate strong model with weak labels
    test_questions = [q.question for q in test_set]

    responses = await get_batch_responses_with_few_shot(
        api=api,
        questions=test_questions,
        few_shot_examples=few_shot_examples,
        system_prompt=system_prompt,
        model=strong_model,
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
    print("Weak Few-Shot Sweep: Varying Number of Examples")
    print("=" * 70)

    # Configuration
    num_shots_list = [0, 5, 10, 25, 50, 100, 200]
    test_size = 200

    # Define weak -> strong model pairs
    # Only same size or larger models
    model_pairs = [
        ("meta-llama/llama-3.1-8b-instruct", "meta-llama/llama-3.1-8b-instruct"),
        ("meta-llama/llama-3.1-8b-instruct", "meta-llama/llama-3.1-70b-instruct"),
        ("meta-llama/llama-3.1-8b-instruct", "qwen/qwen-2.5-72b-instruct"),
        ("meta-llama/llama-3.1-70b-instruct", "meta-llama/llama-3.1-70b-instruct"),
        ("meta-llama/llama-3.1-70b-instruct", "qwen/qwen-2.5-72b-instruct"),
    ]

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
    print(f"\n3. Running sweep over {len(num_shots_list)} shot counts for {len(model_pairs)} model pairs...")
    print(f"   Shot counts: {num_shots_list}\n")

    all_results = {}

    for weak_model, strong_model in model_pairs:
        pair_key = f"{weak_model} -> {strong_model}"
        weak_name = weak_model.split('/')[-1]
        strong_name = strong_model.split('/')[-1]

        print(f"\n   Pair: {weak_name} → {strong_name}")

        pair_results = []

        for num_shots in num_shots_list:
            result = await evaluate_with_weak_labels(
                api=api,
                weak_model=weak_model,
                strong_model=strong_model,
                num_few_shot=num_shots,
                test_set=test_set,
                train_set=train_set,
                system_prompt=system_prompt
            )
            pair_results.append(result)

        all_results[pair_key] = {
            "weak_model": weak_model,
            "weak_model_name": weak_name,
            "strong_model": strong_model,
            "strong_model_name": strong_name,
            "results": pair_results
        }

    # 4. Save results
    # Go up to code/, then up to project root, then to results/
    results_dir = Path(__file__).parent.parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = results_dir / f"weak_sweep_{timestamp}.json"

    output_data = {
        "timestamp": timestamp,
        "experiment": "weak_labels_sweep",
        "num_shots_list": num_shots_list,
        "dataset": {
            "name": "TruthfulQA",
            "test_size": len(test_set),
            "train_size": len(train_set),
        },
        "model_pairs": all_results
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    # 5. Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    for pair_key, data in all_results.items():
        print(f"\n{data['weak_model_name']} → {data['strong_model_name']}:")
        for result in data["results"]:
            num_shots = result["num_few_shot"]
            accuracy = result["accuracy"]
            print(f"  {num_shots:3d} shots: {accuracy:.1%}")

    print("\n" + "=" * 70)
    print(f"Results saved to: {output_file}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
