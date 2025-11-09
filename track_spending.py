#!/usr/bin/env python3
"""Track OpenRouter API spending."""

import csv
from pathlib import Path
from typing import Optional


def get_project_root() -> Path:
    """Get the project root directory."""
    current = Path(__file__).resolve()
    return current.parent


SPENDING_LOG_DIR = get_project_root() / "SPEND"
OPENROUTER_LOG_FILE = SPENDING_LOG_DIR / "openrouter_spending_log.csv"

print(f"SPENDING_LOG_DIR: {SPENDING_LOG_DIR}")
print(f"OPENROUTER_LOG_FILE: {OPENROUTER_LOG_FILE}")

def get_total_spending(log_file: Optional[Path] = None) -> dict:
    """Get total spending from the log file.

    Args:
        log_file: Path to log file. If None, uses default OpenRouter log.

    Returns:
        Dictionary with spending statistics
    """
    if log_file is None:
        log_file = OPENROUTER_LOG_FILE

    if not log_file.exists():
        return {
            "total": 0.0,
            "by_model": {},
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_calls": 0,
            "cache_hits": 0,
        }

    total = 0.0
    by_model = {}
    total_input_tokens = 0
    total_output_tokens = 0
    total_calls = 0
    cache_hits = 0

    with open(log_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cost = float(row['total_cost'])
            total += cost
            total_calls += 1

            # Count cache hits
            if row.get('cache_hit', 'False') == 'True':
                cache_hits += 1

            # By model
            model = row['model']
            if model not in by_model:
                by_model[model] = {
                    "cost": 0.0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "calls": 0,
                    "cache_hits": 0,
                }

            by_model[model]["cost"] += cost
            by_model[model]["input_tokens"] += int(row.get('input_tokens', 0))
            by_model[model]["output_tokens"] += int(row.get('output_tokens', 0))
            by_model[model]["calls"] += 1

            if row.get('cache_hit', 'False') == 'True':
                by_model[model]["cache_hits"] += 1

            total_input_tokens += int(row.get('input_tokens', 0))
            total_output_tokens += int(row.get('output_tokens', 0))

    return {
        "total": total,
        "by_model": by_model,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_calls": total_calls,
        "cache_hits": cache_hits,
    }


def print_spending_summary(log_file: Optional[Path] = None, budget: float = 300.0):
    """Print a formatted summary of OpenRouter API spending.

    Args:
        log_file: Path to log file. If None, uses default OpenRouter log.
        budget: Budget limit in dollars (default: $300)
    """
    spending = get_total_spending(log_file)

    print("\n" + "=" * 70)
    print("OPENROUTER API SPENDING SUMMARY")
    print("=" * 70)

    if spending['total'] == 0:
        print("\nNo API calls logged yet.")
        print(f"Budget: ${budget:.2f}")
        if log_file:
            print(f"Log file: {log_file}")
        else:
            print(f"Log directory: {SPENDING_LOG_DIR}")
        print("=" * 70)
        return

    print(f"\n💰 TOTAL SPENDING: ${spending['total']:.2f}")
    print(f"📊 BUDGET REMAINING: ${budget - spending['total']:.2f} / ${budget:.2f}")

    budget_pct = (spending['total'] / budget * 100) if budget > 0 else 0
    print(f"📈 BUDGET USED: {budget_pct:.1f}%")

    print(f"\n📞 Total API Calls: {spending['total_calls']:,}")
    cache_pct = (spending['cache_hits'] / spending['total_calls'] * 100) if spending['total_calls'] > 0 else 0
    print(f"⚡ Cache Hits: {spending['cache_hits']:,} ({cache_pct:.1f}%)")
    print(f"📝 Total Tokens: {spending['total_input_tokens']:,} input, {spending['total_output_tokens']:,} output")

    # By model
    if spending['by_model']:
        print("\n🤖 BREAKDOWN BY MODEL:")
        print("-" * 70)

        # Sort by cost descending
        sorted_models = sorted(
            spending['by_model'].items(),
            key=lambda x: x[1]['cost'],
            reverse=True
        )

        for model, stats in sorted_models:
            model_name = model.split('/')[-1] if '/' in model else model
            print(f"\n{model_name}:")
            print(f"  💰 Cost: ${stats['cost']:.2f}")
            print(f"  📞 API Calls: {stats['calls']:,}")

            cache_pct = (stats['cache_hits'] / stats['calls'] * 100) if stats['calls'] > 0 else 0
            print(f"  ⚡ Cache Hits: {stats['cache_hits']:,} ({cache_pct:.1f}%)")
            print(f"  📥 Input Tokens: {stats['input_tokens']:,}")
            print(f"  📤 Output Tokens: {stats['output_tokens']:,}")

            if stats['calls'] > 0:
                avg_cost = stats['cost'] / stats['calls']
                avg_input = stats['input_tokens'] / stats['calls']
                avg_output = stats['output_tokens'] / stats['calls']
                print(f"  📊 Avg per call: ${avg_cost:.4f}, {avg_input:.0f} in, {avg_output:.0f} out")

    print("\n" + "=" * 70)
    if log_file:
        print(f"Log file: {log_file}")
    else:
        print(f"Log directory: {SPENDING_LOG_DIR}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    print_spending_summary()
