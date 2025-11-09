"""Helper utilities for TruthfulQA experiments with OpenRouter."""

from .dataset_utils import (
    DatasetQuestion,
    FormattedDatasetQuestion,
    TruthfulDataset,
    load_truthful_qa,
)

from .api_utils import (
    OpenRouterAPI,
    get_few_shot_prompt,
    get_response_with_few_shot,
    get_batch_responses_with_few_shot,
)

from .eval_utils import (
    extract_answer_letter,
    score_response,
    score_batch,
    compute_pgr,
    print_evaluation_results,
    analyze_errors,
)

__all__ = [
    # Dataset utilities
    "DatasetQuestion",
    "FormattedDatasetQuestion",
    "TruthfulDataset",
    "load_truthful_qa",
    # API utilities
    "OpenRouterAPI",
    "get_few_shot_prompt",
    "get_response_with_few_shot",
    "get_batch_responses_with_few_shot",
    # Evaluation utilities
    "extract_answer_letter",
    "score_response",
    "score_batch",
    "compute_pgr",
    "print_evaluation_results",
    "analyze_errors",
]
