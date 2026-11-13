# Support Ticket Agent

An agent that reads support tickets, classifies each one (bug, billing,
feature_request, question, other), sets a priority with a short reason, drafts
replies for simple questions, sends anything uncertain to a human, and logs every
decision.

## How to run

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env      # then add your GEMINI_API_KEY
python src/classifier.py    # classifies data/tickets.csv
python src/evaluate.py      # accuracy on the 10-ticket eval set
python src/cost.py --input-price 0.75 --output-price 3.75   # USD per 1M tokens
```

Outputs: `results/results.csv` (one row per ticket) and `logs/agent_log.jsonl`
(one JSON line per decision, with the ticket text, model, result, escalation
reason and token counts). The first-run files are kept as `results/results_run1.csv`,
`results/eval_report_run1.txt` and `logs/agent_log_run1.jsonl` for comparison.

## My choices

- **Gemini free tier.** It costs nothing for a test like this. Only `call_model()`
  talks to the model, so switching provider means changing one function.
- **Structured output with a pydantic schema.** The model must return valid
  category/priority values; anything invalid is treated as an error, not guessed.
- **Escalation is rule-based, not only the model's say-so.** A ticket goes to a
  human if confidence is below 0.75, the category is `other`, or it mentions a
  sensitive topic (refund, legal, security, unauthorized charge, billing owner,
  cancel my account, ...). Any one rule is enough.
- **Fail safe.** If the API fails or the output is invalid, the ticket is escalated
  with the error recorded. The agent never raises and never guesses.
- **No auto-reply on escalated tickets.** A draft is only kept for tickets the agent
  handles itself.
- **Retries with backoff** for temporary API errors (503, 429), plus a delay between
  calls to stay under the free-tier rate limit.
- **The agent never reads the answer key.** The tickets CSV has an
  `expected_category` column; the code only uses `ticket_id` and `ticket_text`.

## Assumptions

- Tickets are in English, with one issue per ticket, at roughly 200 tickets per week.
- A wrong auto-handled ticket costs more than an extra human review, so the
  thresholds lean towards escalating.
- "Escalate" means the agent should not decide alone: vague tickets, refunds,
  suspected fraud, account ownership changes. A clear bug or billing ticket is
  classified automatically and still reaches the team normally.

## Eval (10 tickets)

I picked 10 tickets (5 clear, 5 tricky) and labeled category, priority and
should-escalate from the ticket text before checking the agent's output. The labels
were drafted with Claude's help and then reviewed by me. They are in
`data/eval_labels.csv`.

| Measure | Run 1 | Run 2 |
|---|---|---|
| Category accuracy | 8/10 | 9/10 |
| Priority accuracy | 9/10 | 9/10 |
| Escalation accuracy | 6/10 | 9/10 |

**Run 1.** The agent never escalated a ticket it should have handled itself, but it
missed 4 of the 5 tickets that should have gone to a human: an unrecognized charge
(T011), a billing owner who left the company (T029), and two vague tickets (T026,
T027) that it classified instead of flagging. The cause: the model's self-reported
confidence was almost always 0.8 to 1.0, so the confidence rule rarely fired, and my
keyword list did not cover these cases.

**Changes after run 1.** I added keywords (`unauthorized`, `unrecognized`,
`don't recognize`, `fraud`, `billing owner`, `account owner`) and two prompt rules:
use `other` with confidence below 0.6 for vague tickets, and keep confidence of
0.9 or higher for clear, specific tickets.

**Run 2.** Escalation went from 6/10 to 9/10. Remaining misses:
- **T026** (an unclear charge of 249): the agent called it plain billing and handled
  it automatically. It still sounds like a billing ticket, so the vague-ticket rule
  did not fire.
- **T027** ("Nothing is working"): escalated correctly, but the agent said `urgent`
  and I labeled `high`. A human sees it either way.

**Caveat.** I chose the fixes after seeing which of these 10 tickets failed, so the
run 2 numbers overstate real performance. With only 10 tickets the results are
indicative, not proof. A fresh set of labeled tickets would be the real test.

## Cost

About **$0.054 per 100 tickets** on the paid tier. From one run of 30 tickets:
11,728 input tokens, 2,009 output tokens and no thinking tokens, priced at $0.75 per
1M input and $3.75 per 1M output (Gemini 3.8 Flash, introductory pricing through
December 31, 2026). `src/cost.py` bills thinking tokens as output. I ran on the free
tier, so my actual spend was $0. This is an estimate from one run of short tickets,
and prices change after the introductory period. At about 200 tickets a week it comes
to roughly $0.11 a week.

## What I would do with more time

- A larger eval set with a fresh holdout, including ambiguous, mixed and non-English
  tickets.
- Calibrate confidence. The model reports high confidence on almost everything, so
  it is a weak escalation signal on its own.
- A review queue for escalated tickets, with the agent's suggested category attached.
- Consider always sending `urgent` tickets to a human.
- Add a rule for tickets that do not say which account or charge they mean (T026).
- Compare a second model on the same eval set.

## Harness setup

TODO: describe what you used (CLAUDE.md, Cursor rules, MCP config) and where it is in the repo.