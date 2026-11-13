"""
Estimate the cost per 100 tickets from the token counts in results/results.csv.

Usage (prices are USD per 1 million tokens, from your model's pricing page):
    python src/cost.py --input-price 0.30 --output-price 2.50

Thinking tokens are billed at the output price. The free tier costs $0, so this
is the estimate for the paid tier.
"""

import argparse
import csv
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results" / "results.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-price", type=float, required=True, help="USD per 1M input tokens")
    parser.add_argument("--output-price", type=float, required=True, help="USD per 1M output tokens")
    args = parser.parse_args()

    with open(RESULTS, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    n = len(rows)
    inp = sum(int(r["input_tokens"]) for r in rows)
    out = sum(int(r["output_tokens"]) for r in rows)
    think = sum(int(r["thinking_tokens"]) for r in rows)

    total = inp / 1e6 * args.input_price + (out + think) / 1e6 * args.output_price
    per_ticket = total / n

    print(f"Tickets: {n}")
    print(f"Tokens: input={inp}, output={out}, thinking={think}")
    print(f"Cost for these {n} tickets: ${total:.4f}")
    print(f"Per ticket: ${per_ticket:.5f}")
    print(f"Per 100 tickets: ${per_ticket * 100:.3f}")


if __name__ == "__main__":
    main()