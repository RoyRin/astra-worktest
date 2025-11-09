"""Dataset loading and formatting utilities for TruthfulQA."""

import random
from abc import ABC, abstractmethod
from typing import List

import pydantic
from datasets import load_dataset


class DatasetQuestion(pydantic.BaseModel):
    """Raw dataset question with correct and incorrect answers."""
    question_id: int
    question: str
    incorrect_answers: List[str]
    correct_answer: str
    solution: str


class FormattedDatasetQuestion(pydantic.BaseModel):
    """Formatted multiple-choice question with letter answer."""
    question_id: int
    question: str  # Question with formatted choices (A, B, C, D...)
    answer: str  # Correct letter (e.g., "A", "B", "C")
    solution: str


class Dataset(ABC):
    """Abstract base class for datasets."""

    def __init__(self, dataset: List[dict]):
        self.dataset = dataset

    @abstractmethod
    def unpack_single(self, row: dict, index: int) -> DatasetQuestion:
        """Convert a raw dataset row to a DatasetQuestion."""
        pass

    def convert_to_questions(self, dataset: List[dict]) -> List[DatasetQuestion]:
        """Convert entire dataset to list of DatasetQuestion objects."""
        return [self.unpack_single(row, i) for i, row in enumerate(dataset)]

    def format_row(self, item: DatasetQuestion, seed: int = 42) -> FormattedDatasetQuestion:
        """Format a DatasetQuestion into multiple-choice format with shuffled options.

        Args:
            item: DatasetQuestion to format
            seed: Random seed (default: 42)

        Returns:
            FormattedDatasetQuestion with letter-based answer choices
        """
        question_id = item.question_id
        question = item.question
        correct_answer = item.correct_answer
        incorrect_answers = item.incorrect_answers
        solution = item.solution

        assert correct_answer not in incorrect_answers, f"{correct_answer} in {incorrect_answers}"
        choices = [correct_answer] + incorrect_answers

        # Shuffle choices with deterministic seed
        random.seed(question_id + seed)
        random.shuffle(choices)

        # Format as A: choice1, B: choice2, etc.
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        letters = letters[:len(choices)]
        choices_str = [f"{letters[i]}) {choice}" for i, choice in enumerate(choices)]
        choices_str = "\n".join(choices_str)
        correct_letter = letters[choices.index(correct_answer)]

        question = f"{question}\n\n{choices_str}"

        return FormattedDatasetQuestion(
            question_id=question_id,
            question=question,
            answer=correct_letter,
            solution=solution,
        )

    def format_dataset(self) -> List[FormattedDatasetQuestion]:
        """Format all questions in the dataset."""
        return [self.format_row(q) for q in self.dataset]

    def filter_questions(self, questions_to_remove: set[str]):
        """Remove specific questions from the dataset."""
        self.dataset = [q for q in self.dataset if q.question not in questions_to_remove]


class TruthfulDataset(Dataset):
    """TruthfulQA dataset loader and formatter."""

    def __init__(self, dataset_split: str = "validation"):
        """Load TruthfulQA dataset from HuggingFace.

        Args:
            dataset_split: Which split to load (default: "validation")
        """
        dataset = load_dataset("EleutherAI/truthful_qa_mc", "multiple_choice")
        hf_dataset = dataset[dataset_split]
        self.dataset = self.convert_to_questions(hf_dataset)

    @staticmethod
    def raw_to_question(raw):
        """Extract question text from raw dataset entry."""
        sentence = raw["question"].rstrip("\n")
        base_question = f"""{sentence}"""
        return base_question

    def unpack_single(self, item: dict, index: int) -> DatasetQuestion:
        """Convert a raw TruthfulQA entry to a DatasetQuestion."""
        question = self.raw_to_question(item)
        answer_key = item["label"]
        correct_answer = item["choices"][answer_key]

        incorrect_answers = [
            item["choices"][i]
            for i in range(len(item["choices"]))
            if i != answer_key
        ]

        if correct_answer in incorrect_answers:
            incorrect_answers.remove(correct_answer)

        return DatasetQuestion(
            question_id=index,
            question=question,
            incorrect_answers=incorrect_answers,
            correct_answer=correct_answer,
            solution=""
        )


def load_truthful_qa(
    dataset_split: str = "validation",
    test_size: int = 200,
    random_seed: int = 42
) -> tuple[List[FormattedDatasetQuestion], List[FormattedDatasetQuestion]]:
    """Load and split TruthfulQA dataset into train and test sets.

    Args:
        dataset_split: HuggingFace dataset split to use (default: "validation")
        test_size: Number of questions for test set (default: 200)
        random_seed: Random seed for shuffling (default: 42)

    Returns:
        Tuple of (test_questions, train_questions)
    """
    # Load and format dataset
    truthful_dataset = TruthfulDataset(dataset_split=dataset_split)
    formatted_truthful = truthful_dataset.format_dataset()

    # Shuffle and split
    random.seed(random_seed)
    truthful_all = random.sample(formatted_truthful, len(formatted_truthful))
    truthful_test = truthful_all[:test_size]
    truthful_train = truthful_all[test_size:]

    return truthful_test, truthful_train


if __name__ == "__main__":
    # Example usage
    print("Loading TruthfulQA dataset...")
    test_set, train_set = load_truthful_qa()

    print(f"\nDataset loaded:")
    print(f"  Test set: {len(test_set)} questions")
    print(f"  Train set: {len(train_set)} questions")

    print(f"\nExample question:")
    example = test_set[0]
    print(f"  Question ID: {example.question_id}")
    print(f"  Answer: {example.answer}")
    print(f"\n{example.question}")
