"""
Support ticket agent.

Reads support tickets, classifies each one (bug, billing, feature_request,
question, other), assigns a priority with a short reason, drafts a reply for
simple questions, escalates to a human when unsure, and logs every decision.

Model provider: Google Gemini (free tier). Only call_model() talks to the
model, so switching provider later means changing that one function.
"""

import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

TICKETS_CSV = PROJECT_ROOT / "data" / "tickets.csv"
LOG_FILE = PROJECT_ROOT / "logs" / "agent_log.jsonl"
RESULTS_CSV = PROJECT_ROOT / "results" / "results.csv"

CONFIDENCE_THRESHOLD = 0.75
SECONDS_BETWEEN_CALLS = 5  # keeps us under the free-tier rate limit
MAX_ATTEMPTS = 4           # retries for temporary API errors (503, 429)

# Topics a human should always see, whatever the model says.
SENSITIVE_KEYWORDS = [
    "refund", "chargeback", "lawyer", "legal", "sue ", "gdpr",
    "security", "breach", "hacked", "data loss", "cancel my account",
    # added after the first eval run (missed T011 and T029):
    "unauthorized", "unrecognized", "don't recognize", "fraud",
    "billing owner", "account owner",
]

SYSTEM_PROMPT = """You are a support ticket triage agent for a software company.

Read the ticket and fill in these fields:

- category: one of "bug", "billing", "feature_request", "question", "other"
- priority: one of "low", "medium", "high", "urgent"
- priority_reason: one short sentence explaining the priority
- confidence: a number from 0 to 1 for how sure you are about the category
- reply_draft: a short, polite reply if the ticket is a simple question you can
  answer. Otherwise null.

Priority guidance:
- urgent: outage, data loss, security issue, or a customer who cannot use the product at all
- high: a core feature is broken, or a billing error affects money
- medium: a problem with a workaround, or a time-sensitive question
- low: general questions, feature ideas, minor annoyances

Rules:
- Only draft a reply for simple, low-risk questions (for example password reset or how-to).
- Never promise refunds, discounts, or timelines in a draft reply.
- If the ticket is ambiguous, mixed, or does not fit a category, lower your
  confidence instead of guessing.
- If the ticket is too vague to tell what is wrong (for example "nothing is
  working" with no detail), or key information is missing (for example it is
  unclear which account or charge is meant), use category "other" and a
  confidence below 0.6. Do not pick "bug" or "billing" just because it sounds close.
- Confidence must reflect real uncertainty. Use 0.9 or higher only when the
  ticket is clear and specific.
"""


class TicketDecision(BaseModel):
    category: Literal["bug", "billing", "feature_request", "question", "other"]
    priority: Literal["low", "medium", "high", "urgent"]
    priority_reason: str
    confidence: float
    reply_draft: Optional[str] = None


def call_model(ticket_text: str):
    """Call Gemini, retrying when the service is overloaded or rate limited."""
    delay = 5
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return client.models.generate_content(
                model=MODEL,
                contents=f"Classify this ticket.\n\nTicket:\n{ticket_text}",
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_schema=TicketDecision,
                    temperature=0,
                ),
            )
        except Exception as error:
            message = str(error)
            temporary = any(
                marker in message
                for marker in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED")
            )
            if temporary and attempt < MAX_ATTEMPTS:
                print(
                    f"    temporary API error, retrying in {delay}s "
                    f"(attempt {attempt}/{MAX_ATTEMPTS})"
                )
                time.sleep(delay)
                delay *= 2  # waits 5s, 10s, 20s
                continue
            raise


def decide_escalation(ticket_text: str, result: dict) -> tuple[bool, str]:
    """Return (escalate, reason). Any single rule is enough to escalate."""
    reasons = []

    if result["confidence"] < CONFIDENCE_THRESHOLD:
        reasons.append(f"low confidence ({result['confidence']:.2f})")

    if result["category"] == "other":
        reasons.append("category is 'other'")

    lowered = ticket_text.lower()
    hits = [word.strip() for word in SENSITIVE_KEYWORDS if word in lowered]
    if hits:
        reasons.append("sensitive topic: " + ", ".join(hits))

    return (len(reasons) > 0, "; ".join(reasons))


def classify_ticket(ticket_id: str, ticket_text: str) -> dict:
    """Classify one ticket. Never raises: failures are escalated to a human."""
    try:
        if not ticket_text.strip():
            raise ValueError("ticket text is empty")
        response = call_model(ticket_text)

        decision = TicketDecision.model_validate_json(response.text)
        result = decision.model_dump()
        result["confidence"] = max(0.0, min(1.0, result["confidence"]))

        usage = response.usage_metadata
        input_tokens = usage.prompt_token_count or 0
        output_tokens = usage.candidates_token_count or 0
        thinking_tokens = usage.thoughts_token_count or 0

    except Exception as error:  # API failure or invalid output: do not guess
        return {
            "ticket_id": ticket_id,
            "category": "other",
            "priority": "medium",
            "priority_reason": "Agent could not classify this ticket.",
            "confidence": 0.0,
            "reply_draft": None,
            "escalate_to_human": True,
            "escalation_reason": f"agent error: {type(error).__name__}: {error}",
            "input_tokens": 0,
            "output_tokens": 0,
            "thinking_tokens": 0,
        }

    escalate, reason = decide_escalation(ticket_text, result)

    # Never send an auto-drafted reply on a ticket that goes to a human.
    if escalate:
        result["reply_draft"] = None

    return {
        "ticket_id": ticket_id,
        **result,
        "escalate_to_human": escalate,
        "escalation_reason": reason,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "thinking_tokens": thinking_tokens,
    }


def log_decision(ticket_text: str, outcome: dict) -> None:
    """Append one JSON line per ticket: what the agent did and why."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "ticket_text": ticket_text,
        **outcome,
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_tickets(path: Path) -> list[tuple[str, str]]:
    """Read the CSV. Only ticket_id and ticket_text are used, never the answer key."""
    tickets = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for index, row in enumerate(reader, start=1):
            ticket_id = (row.get("ticket_id") or f"T{index:03d}").strip()
            text = (row.get("ticket_text") or "").strip()
            tickets.append((ticket_id, text))
    return tickets


def run_all() -> None:
    tickets = read_tickets(TICKETS_CSV)
    print(f"Loaded {len(tickets)} tickets from {TICKETS_CSV}")

    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for number, (ticket_id, text) in enumerate(tickets, start=1):
        outcome = classify_ticket(ticket_id, text)
        log_decision(text, outcome)
        rows.append(outcome)

        flag = "HUMAN" if outcome["escalate_to_human"] else "auto "
        print(
            f"[{number}/{len(tickets)}] {ticket_id} | {outcome['category']:<15} | "
            f"{outcome['priority']:<6} | conf {outcome['confidence']:.2f} | {flag}"
        )
        if outcome["escalation_reason"].startswith("agent error"):
            print("    ->", outcome["escalation_reason"][:200])

        if number < len(tickets):
            time.sleep(SECONDS_BETWEEN_CALLS)

    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    escalated = sum(1 for row in rows if row["escalate_to_human"])
    print(f"\nDone. {escalated} of {len(rows)} tickets sent to a human.")
    print(f"Results: {RESULTS_CSV}")
    print(f"Log:     {LOG_FILE}")


if __name__ == "__main__":
    print("Starting Support Ticket Agent...")

    if TICKETS_CSV.exists():
        run_all()
    else:
        print(f"{TICKETS_CSV} not found, running one sample ticket instead.\n")
        sample = classify_ticket("TEST001", "How can I reset my account password?")
        print(json.dumps(sample, indent=2))