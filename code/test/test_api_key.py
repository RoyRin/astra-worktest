#!/usr/bin/env python3
"""Test an OpenRouter API key with a simple call (no caching)."""

import argparse
import asyncio
from pathlib import Path

from openai import AsyncOpenAI


async def test_api_key(api_key_path: Path, model: str = "meta-llama/llama-3.1-8b-instruct"):
    """Test an API key by making a simple API call.

    Args:
        api_key_path: Path to file containing API key
        model: Model to test with
    """
    print("=" * 70)
    print("OpenRouter API Key Test")
    print("=" * 70)

    # Read API key
    if not api_key_path.exists():
        print(f"\nError: API key file not found: {api_key_path}")
        return False

    with open(api_key_path, 'r') as f:
        api_key = f.read().strip()

    print(f"\nAPI Key File: {api_key_path}")
    print(f"API Key: {api_key[:20]}...{api_key[-4:]}")  # Show partial key
    print(f"Model: {model}")

    # Create client (no caching)
    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    # Make a simple test call
    print("\nMaking test API call...")

    test_prompt = "What is 2+2? Answer in one word."

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": test_prompt}
            ],
            temperature=0.0,
            max_tokens=10
        )

        # Extract response
        completion = response.choices[0].message.content

        print("\n✓ API call successful!")
        print(f"\nPrompt: {test_prompt}")
        print(f"Response: {completion}")

        # Print usage info if available
        if hasattr(response, 'usage') and response.usage:
            print(f"\nToken usage:")
            print(f"  Input:  {response.usage.prompt_tokens}")
            print(f"  Output: {response.usage.completion_tokens}")
            print(f"  Total:  {response.usage.total_tokens}")

        print("\n" + "=" * 70)
        print("API key is valid and working!")
        print("=" * 70 + "\n")

        return True

    except Exception as e:
        print(f"\n✗ API call failed!")
        print(f"\nError type: {type(e).__name__}")
        print(f"Error message: {str(e)}")

        print("\n" + "=" * 70)
        print("API key test FAILED")
        print("=" * 70 + "\n")

        # Print common issues
        print("Common issues:")
        print("  - API key is invalid or expired")
        print("  - API key has no credits remaining")
        print("  - Network connectivity issues")
        print("  - OpenRouter service is down")

        return False


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test an OpenRouter API key")
    parser.add_argument(
        "api_key_path",
        type=str,
        help="Path to file containing API key"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/llama-3.1-8b-instruct",
        help="Model to test with (default: meta-llama/llama-3.1-8b-instruct)"
    )

    args = parser.parse_args()

    api_key_path = Path(args.api_key_path)

    success = await test_api_key(api_key_path, args.model)

    if not success:
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())
