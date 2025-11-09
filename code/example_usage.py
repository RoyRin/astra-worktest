"""Example usage of the helper utilities."""

import asyncio
from pathlib import Path

from dataset_utils import load_truthful_qa
from api_utils import OpenRouterAPI, get_batch_responses_with_few_shot
from eval_utils import score_batch, print_evaluation_results, compute_pgr

async def main():
    """Example workflow for TruthfulQA experiments comparing three Llama 3.1 models."""

    print("=" * 70)
    print("EXAMPLE: TruthfulQA with Few-Shot Learning - Model Comparison")
    print("=" * 70)

    # 1. Load dataset
    print("\n1. Loading TruthfulQA dataset...")
    test_set, train_set = load_truthful_qa(test_size=20)  # Small test for example
    print(f"   Loaded {len(test_set)} test questions, {len(train_set)} train questions")

    # 2. Initialize API
    print("\n2. Initializing OpenRouter API...")
    api = OpenRouterAPI(
        cache_dir=Path("./cache"),
        num_threads=10
    )

    # 3. Create few-shot examples (using gold labels from train set)
    print("\n3. Creating few-shot examples...")
    num_shots = 5
    few_shot_examples = [
        (train_set[i].question, train_set[i].answer)
        for i in range(num_shots)
    ]
    print(f"   Using {num_shots} few-shot examples")

    # 4. Define models to test
    models = [
        "meta-llama/llama-3.1-8b-instruct",   # ~$3 / 1M tokens
        "meta-llama/llama-3.1-70b-instruct",  # ~$0.8 / 1M tokens
        "meta-llama/llama-3.1-405b-instruct"  # ~$0.2 / 1M tokens
    ]

    system_prompt = "You are a helpful assistant. Answer the following multiple choice question by selecting the correct letter (A, B, C, D, etc.)."
    test_questions = [q.question for q in test_set]

    # 5. Test each model
    print("\n4. Testing models with gold few-shot labels...")
    all_results = {}

    for model in models:
        print(f"\n   Testing {model}...")

        responses = await get_batch_responses_with_few_shot(
            api=api,
            questions=test_questions,
            few_shot_examples=few_shot_examples,
            system_prompt=system_prompt,
            model=model,
            temperature=0.0,
            max_tokens=100
        )

        results = score_batch(responses, test_set)
        all_results[model] = results
        print(f"   ✓ {model.split('/')[-1]}: {results['accuracy']:.1%} ({results['num_correct']}/{results['num_total']})")

    # 6. Summary comparison
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    for model in models:
        results = all_results[model]
        model_name = model.split('/')[-1]
        print(f"\n{model_name}:")
        print(f"  Accuracy: {results['accuracy']:.1%} ({results['num_correct']}/{results['num_total']})")

if __name__ == "__main__":
    # Show PGR calculation example
    #example_pgr_calculation()

    # Run async example (commented out by default)
    # print("\n" * 2)
    asyncio.run(main())
