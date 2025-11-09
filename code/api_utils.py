"""API utilities for making inference calls to OpenRouter models."""

import asyncio
import csv
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt, LLMResponse


# Pricing per 1M tokens for OpenRouter models
OPENROUTER_PRICING = {
    "meta-llama/llama-3.1-405b-instruct": {"input": 0.2, "output": 0.2},
    "meta-llama/llama-3.1-70b-instruct": {"input": 0.8, "output": 0.8},
    "meta-llama/llama-3.1-8b-instruct": {"input": 3.0, "output": 3.0},
}


def get_project_root() -> Path:
    """Get the project root directory."""
    current = Path(__file__).resolve()
    # Go up one level from code/api_utils.py to project root
    return current.parent.parent


# Spending log directory
SPENDING_LOG_DIR = get_project_root() / "SPEND"
SPENDING_LOG_DIR.mkdir(exist_ok=True)
OPENROUTER_LOG_FILE = SPENDING_LOG_DIR / "openrouter_spending_log.csv"


def init_spending_log(log_file: Path):
    """Initialize the spending log CSV file if it doesn't exist."""
    if not log_file.exists():
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'model', 'input_tokens', 'output_tokens',
                'input_cost', 'output_cost', 'total_cost',
                'prompt_preview', 'response_preview', 'cache_hit'
            ])


def log_openrouter_call(
    model: str,
    input_tokens: int,
    output_tokens: int,
    prompt: str = "",
    response: str = "",
    cache_hit: bool = False,
    log_file: Optional[Path] = None
) -> float:
    """Log OpenRouter API call to CSV file.

    Args:
        model: Model name
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        prompt: The prompt sent (optional)
        response: The response received (optional)
        cache_hit: Whether this was a cache hit
        log_file: Custom log file path (optional)

    Returns:
        Total cost for this API call
    """
    if log_file is None:
        log_file = OPENROUTER_LOG_FILE

    # Initialize log if needed
    init_spending_log(log_file)

    # Get pricing
    pricing = OPENROUTER_PRICING.get(model, {"input": 0, "output": 0})

    # Calculate costs (pricing is per 1M tokens)
    # Cache hits cost nothing
    if cache_hit:
        input_cost = 0.0
        output_cost = 0.0
    else:
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]

    total_cost = input_cost + output_cost

    # Prepare previews (first 100 chars)
    prompt_preview = prompt[:100].replace('\n', ' ').replace(',', ';') if prompt else ""
    response_preview = response[:100].replace('\n', ' ').replace(',', ';') if response else ""

    # Prepare row data
    row_data = [
        datetime.now().isoformat(),
        model,
        input_tokens,
        output_tokens,
        f"{input_cost:.6f}",
        f"{output_cost:.6f}",
        f"{total_cost:.6f}",
        prompt_preview,
        response_preview,
        cache_hit
    ]

    # Log to file
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row_data)

    return total_cost


class OpenRouterAPI:
    """Wrapper for OpenRouter API calls with caching."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: Path = Path("./cache"),
        num_threads: int = 50
    ):
        """Initialize OpenRouter API client.

        Args:
            api_key: OpenRouter API key (if None, reads from SECRETS/api.key or OPENROUTER_API_KEY env var)
            cache_dir: Directory for caching responses (default: ./cache)
            num_threads: Max parallel requests (default: 50)
        """
        # Set up API key
        if api_key:
            os.environ["OPENROUTER_API_KEY"] = api_key
        elif "OPENROUTER_API_KEY" not in os.environ:
            # Try to read from SECRETS/api.key
            api_key_file = Path(__file__).parent.parent / "SECRETS" / "api.key"
            if api_key_file.exists():
                api_key = api_key_file.read_text().strip()
                os.environ["OPENROUTER_API_KEY"] = api_key
            else:
                raise ValueError(
                    "No API key provided. Please either:\n"
                    "1. Pass api_key parameter to OpenRouterAPI()\n"
                    "2. Set OPENROUTER_API_KEY environment variable\n"
                    "3. Create SECRETS/api.key file with your API key"
                )

        # Safety-tooling assumes OPENAI_API_KEY is set
        if "OPENAI_API_KEY" not in os.environ:
            os.environ["OPENAI_API_KEY"] = "dummy"

        # Create cache directory if it doesn't exist
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize API
        self.api = InferenceAPI(
            cache_dir=cache_dir,
            openrouter_num_threads=num_threads
        )
        self.semaphore = asyncio.Semaphore(num_threads)

    async def call(
        self,
        model_id: str,
        prompt: Prompt,
        max_attempts_per_api_call: int = 5,
        temperature: float = 0.0,
        max_tokens: int = 500,
        use_cache: bool = True,
        print_prompt_and_response: bool = False,
        **kwargs
    ) -> LLMResponse:
        """Make a single API call to OpenRouter.

        Args:
            model_id: Model identifier (e.g., "meta-llama/llama-3.1-8b-instruct")
            prompt: Prompt object with messages
            max_attempts_per_api_call: Max retry attempts (default: 5)
            temperature: Sampling temperature (default: 0.0)
            max_tokens: Max tokens to generate (default: 500)
            use_cache: Whether to use cached responses (default: True)
            print_prompt_and_response: Print prompt/response (default: False)
            **kwargs: Additional arguments passed to API

        Returns:
            LLMResponse object
        """
        responses = await self.api.__call__(
            model_id=model_id,
            prompt=prompt,
            max_attempts_per_api_call=max_attempts_per_api_call,
            force_provider="openrouter",
            temperature=temperature,
            max_tokens=max_tokens,
            use_cache=use_cache,
            print_prompt_and_response=print_prompt_and_response,
            n=1,
            **kwargs
        )
        response = responses[0]

        # Log spending
        try:
            # Extract token counts from response
            input_tokens = 0
            output_tokens = 0

            # Try to get actual token counts from API response
            if hasattr(response, 'raw_responses') and response.raw_responses:
                raw = response.raw_responses[0]
                if hasattr(raw, 'usage'):
                    input_tokens = raw.usage.get('prompt_tokens', 0)
                    output_tokens = raw.usage.get('completion_tokens', 0)

            # If we couldn't get token counts, estimate them
            # Rough approximation: 1 token ≈ 0.75 words (or ~4 chars)
            if input_tokens == 0 and output_tokens == 0:
                prompt_str = str(prompt)
                response_str = response.completion if hasattr(response, 'completion') else ""

                # Estimate tokens: ~4 characters per token
                input_tokens = max(len(prompt_str) // 4, 1)
                output_tokens = max(len(response_str) // 4, 1)

            # Always log with costs (don't check cache_hit for costs)
            # This gives you the "what if" cost even for cached responses
            log_openrouter_call(
                model=model_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                prompt=str(prompt)[:200],  # First 200 chars
                response=response.completion[:200] if hasattr(response, 'completion') else "",
                cache_hit=False  # Always calculate costs
            )
        except Exception as e:
            # Don't fail if logging fails
            print(f"Warning: Could not log spending: {e}")

        return response


def get_few_shot_prompt(prompts_and_responses: List[tuple[str, str]]) -> List[dict]:
    """Format few-shot examples into alternating user and assistant messages.

    Args:
        prompts_and_responses: List of (prompt, response) tuples

    Returns:
        List of message dictionaries with role and content
    """
    messages = []
    for p, r in prompts_and_responses:
        messages.append({
            "role": "user",
            "content": p,
        })
        messages.append({
            "role": "assistant",
            "content": r
        })
    return messages


async def get_response_with_few_shot(
    api: OpenRouterAPI,
    question: str,
    few_shot_examples: List[tuple[str, str]],
    system_prompt: str,
    model: str,
    max_retries: int = 5,
    max_tokens: int = 500,
    temperature: float = 0.0,
    verbose: bool = False,
    **kwargs
) -> LLMResponse:
    """Get a single response with few-shot prompting.

    Args:
        api: OpenRouterAPI instance
        question: The question to ask
        few_shot_examples: List of (question, answer) tuples for few-shot examples
        system_prompt: System prompt
        model: Model identifier
        max_retries: Max retry attempts (default: 5)
        max_tokens: Max tokens to generate (default: 500)
        temperature: Sampling temperature (default: 0.0)
        verbose: Print timing info (default: False)
        **kwargs: Additional arguments passed to API

    Returns:
        LLMResponse object
    """
    # Build messages
    few_shot_messages = get_few_shot_prompt(few_shot_examples)

    system_message = [{"role": "system", "content": system_prompt}]
    user_message = [{"role": "user", "content": question}]

    messages = system_message + few_shot_messages + user_message
    prompt = Prompt(messages=messages)

    # Make API call
    async with api.semaphore:
        response = await api.call(
            model_id=model,
            prompt=prompt,
            max_attempts_per_api_call=max_retries,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )

        if verbose:
            print(f"Got response from {model} after {response.duration:.2f}s")

        return response


async def get_batch_responses_with_few_shot(
    api: OpenRouterAPI,
    questions: List[str],
    few_shot_examples: List[tuple[str, str]],
    system_prompt: str,
    model: str,
    **kwargs
) -> List[LLMResponse]:
    """Get responses for multiple questions with the same few-shot examples.

    Args:
        api: OpenRouterAPI instance
        questions: List of questions
        few_shot_examples: List of (question, answer) tuples for few-shot examples
        system_prompt: System prompt
        model: Model identifier
        **kwargs: Additional arguments passed to get_response_with_few_shot

    Returns:
        List of LLMResponse objects
    """
    responses = await asyncio.gather(
        *[
            get_response_with_few_shot(
                api=api,
                question=q,
                few_shot_examples=few_shot_examples,
                system_prompt=system_prompt,
                model=model,
                **kwargs
            )
            for q in questions
        ]
    )
    return responses


if __name__ == "__main__":
    # Example usage
    import asyncio

    async def main():
        # Initialize API
        api = OpenRouterAPI(cache_dir=Path("./cache"))

        # Example few-shot examples
        few_shot = [
            ("What is 2 + 2?", "2 + 2 = 4."),
            ("What is 49*7?", "49 * 7 = 343.")
        ]

        # Get a single response
        response = await get_response_with_few_shot(
            api=api,
            question="What is 64 ** 2?",
            few_shot_examples=few_shot,
            system_prompt="You are a math expert.",
            model="meta-llama/llama-3.1-8b-instruct",
            verbose=True
        )

        print(f"\nResponse: {response.completion}")

    asyncio.run(main())
