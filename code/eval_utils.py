"""Evaluation utilities for scoring model responses."""

import re
from typing import List, Dict, Any
import numpy as np
from safetytooling.data_models import LLMResponse

try:
    from .dataset_utils import FormattedDatasetQuestion
except ImportError:
    from dataset_utils import FormattedDatasetQuestion


def extract_answer_letter(response: str) -> str:
    """Extract the answer letter (A, B, C, D, etc.) from a model response.

    Args:
        response: Model response text

    Returns:
        Single letter answer (A-Z), or empty string if not found
    """
    # Clean up response
    response = response.strip()

    # Pattern 1: Look for "Answer: X" or "The answer is X"
    patterns = [
        r'(?:answer|Answer|ANSWER)(?:\s+is)?(?:\s*:)?\s*([A-Z])',
        r'^([A-Z])(?:\)|\.|\s|$)',  # Letter at start with delimiter
        r'\b([A-Z])\)(?:\s|$)',  # Letter with closing paren
    ]

    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            return match.group(1)

    # Pattern 2: If response is just a single letter
    if len(response) == 1 and response.isalpha() and response.isupper():
        return response

    # Pattern 3: Take the first capital letter that appears
    for char in response:
        if char.isupper() and char.isalpha():
            return char

    return ""


def score_response(response: str, ground_truth: str) -> bool:
    """Score a single response against ground truth.

    Args:
        response: Model response text
        ground_truth: Correct answer letter (e.g., "A", "B", "C")

    Returns:
        True if correct, False otherwise
    """
    predicted = extract_answer_letter(response)
    return predicted == ground_truth


def score_batch(
    responses: List[LLMResponse],
    questions: List[FormattedDatasetQuestion]
) -> Dict[str, Any]:
    """Score a batch of responses.

    Args:
        responses: List of LLMResponse objects
        questions: List of FormattedDatasetQuestion objects (same length as responses)

    Returns:
        Dictionary with:
            - accuracy: Float accuracy (0-1)
            - num_correct: Number of correct answers
            - num_total: Total number of questions
            - predictions: List of predicted letters
            - correct: List of booleans indicating correctness
    """
    assert len(responses) == len(questions), "Responses and questions must have same length"

    predictions = []
    correct = []

    for response, question in zip(responses, questions):
        predicted = extract_answer_letter(response.completion)
        is_correct = predicted == question.answer

        predictions.append(predicted)
        correct.append(is_correct)

    num_correct = sum(correct)
    num_total = len(correct)
    accuracy = num_correct / num_total if num_total > 0 else 0.0

    return {
        "accuracy": accuracy,
        "num_correct": num_correct,
        "num_total": num_total,
        "predictions": predictions,
        "correct": correct
    }


def compute_pgr(
    weak_accuracy: float,
    strong_with_weak_labels_accuracy: float,
    strong_with_gold_labels_accuracy: float
) -> float:
    """Compute Performance Gap Recovered (PGR) metric.

    PGR measures how much of the performance gap between weak and strong models
    is recovered when the strong model is given weak labels vs gold labels.

    PGR = (strong_weak - weak) / (strong_gold - weak)

    Args:
        weak_accuracy: Weak model accuracy with gold labels
        strong_with_weak_labels_accuracy: Strong model accuracy with weak labels
        strong_with_gold_labels_accuracy: Strong model accuracy with gold labels

    Returns:
        PGR value (can be negative if strong model does worse with weak labels)
    """
    numerator = strong_with_weak_labels_accuracy - weak_accuracy
    denominator = strong_with_gold_labels_accuracy - weak_accuracy

    if denominator == 0:
        return 0.0

    return numerator / denominator


def print_evaluation_results(
    results: Dict[str, Any],
    model_name: str = "",
    condition: str = ""
):
    """Print formatted evaluation results.

    Args:
        results: Results dictionary from score_batch
        model_name: Name of the model (optional)
        condition: Experimental condition (optional)
    """
    print("\n" + "=" * 60)
    if model_name:
        print(f"Model: {model_name}")
    if condition:
        print(f"Condition: {condition}")
    print("=" * 60)
    print(f"Accuracy: {results['accuracy']:.2%} ({results['num_correct']}/{results['num_total']})")
    print("=" * 60)


def analyze_errors(
    responses: List[LLMResponse],
    questions: List[FormattedDatasetQuestion],
    max_examples: int = 5
) -> Dict[str, Any]:
    """Analyze errors in model predictions.

    Args:
        responses: List of LLMResponse objects
        questions: List of FormattedDatasetQuestion objects
        max_examples: Max number of error examples to return (default: 5)

    Returns:
        Dictionary with error analysis
    """
    errors = []

    for response, question in zip(responses, questions):
        predicted = extract_answer_letter(response.completion)
        if predicted != question.answer:
            errors.append({
                "question_id": question.question_id,
                "question": question.question,
                "predicted": predicted,
                "correct": question.answer,
                "response": response.completion
            })

    return {
        "num_errors": len(errors),
        "error_rate": len(errors) / len(questions) if questions else 0,
        "error_examples": errors[:max_examples]
    }


if __name__ == "__main__":
    # Example usage
    try:
        from .dataset_utils import load_truthful_qa
    except ImportError:
        from dataset_utils import load_truthful_qa

    # Load dataset
    test_set, train_set = load_truthful_qa()

    # Simulate some predictions
    print("Testing evaluation utilities...")

    # Test answer extraction
    test_responses = [
        "A) is the correct answer",
        "The answer is B",
        "I think C",
        "Answer: D",
        "E",
    ]

    print("\nTesting answer extraction:")
    for resp in test_responses:
        letter = extract_answer_letter(resp)
        print(f"  '{resp}' -> '{letter}'")

    # Test scoring
    print("\nTesting PGR calculation:")
    weak_acc = 0.5
    strong_weak = 0.7
    strong_gold = 0.8

    pgr = compute_pgr(weak_acc, strong_weak, strong_gold)
    print(f"  Weak model (gold labels): {weak_acc:.2%}")
    print(f"  Strong model (weak labels): {strong_weak:.2%}")
    print(f"  Strong model (gold labels): {strong_gold:.2%}")
    print(f"  PGR: {pgr:.2%}")
