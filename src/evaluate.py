"""
Eval: compare the agent's output with my hand labels on the 10-ticket eval set.

Uses results/results.csv (from `python src/classifier.py`), so it makes no
extra API calls.

Labels file: data/eval_labels.csv with columns
    ticket_id, category, priority, should_escalate
should_escalate is yes/no. Label the tickets BEFORE looking at the agent's output.
"""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "data" / "eval_labels.csv"
RESULTS = ROOT / "results" / "results.csv"
REPORT = ROOT / "results" / "eval_report.txt"


def load(path: Path) -> dict:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return {row["ticket_id"].strip(): row for row in csv.DictReader(f)}


def to_bool(value: str) -> bool:
    return str(value).strip().lower() in ("true", "yes", "y", "1")


def main() -> None:
    labels = load(LABELS)
    results = load(RESULTS)

    missing = [tid for tid in labels if tid not in results]
    if missing:
        raise SystemExit(f"Not in results.csv (run the agent first): {missing}")

    totals = {"category": 0, "priority": 0, "escalation": 0}
    lines = []

    for tid, label in labels.items():
        got = results[tid]
        checks = {
            "category": label["category"].strip() == got["category"].strip(),
            "priority": label["priority"].strip() == got["priority"].strip(),
            "escalation": to_bool(label["should_escalate"]) == to_bool(got["escalate_to_human"]),
        }
        for name, ok in checks.items():
            totals[name] += ok

        if not all(checks.values()):
            wrong = [name for name, ok in checks.items() if not ok]
            lines.append(
                f"{tid}: wrong on {', '.join(wrong)} | "
                f"mine: {label['category']}/{label['priority']}/escalate={label['should_escalate']} | "
                f"agent: {got['category']}/{got['priority']}/escalate={got['escalate_to_human']}"
            )

    n = len(labels)
    report = [
        f"Eval set: {n} tickets",
        f"Category accuracy:   {totals['category']}/{n} ({totals['category'] / n:.0%})",
        f"Priority accuracy:   {totals['priority']}/{n} ({totals['priority'] / n:.0%})",
        f"Escalation accuracy: {totals['escalation']}/{n} ({totals['escalation'] / n:.0%})",
        "",
        "Misses:" if lines else "No misses.",
        *lines,
    ]
    text = "\n".join(report)
    print(text)
    REPORT.write_text(text + "\n", encoding="utf-8")
    print(f"\nSaved to {REPORT}")


if __name__ == "__main__":
    main()