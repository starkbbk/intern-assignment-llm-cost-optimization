# ShopAssist — support copilot cost lab

**ShopAssist** is a Tier-1 support agent for an e-commerce company. For each ticket it:

1. Reads the customer thread (`data/tickets/`)
2. Loads order facts (`data/orders/` via `shared/tools.py`)
3. Picks an action (`REFUND`, `REPLACE`, `SEND_ARTICLE`, `ESCALATE_TIER2`)
4. Cites the help-center article that backs the decision (`data/kb/`)

| Track | Code | Spec |
|-------|------|------|
| Q1 — single call | [question1/agent.py](question1/agent.py) | [questions/question-1.md](questions/question-1.md) |
| Q2 — pipeline | [question2/agent.py](question2/agent.py) | [questions/question-2.md](questions/question-2.md) |

## Optimization submission

The optimized Q1 and Q2 agents preserve routing for all five available tickets.
Public costs are **1704–1905 points per ticket for Q1** (target ≤3000) and
**1245–1515 for Q2** (target ≤2500), with no cached-input discount.

See [SUBMISSION.md](SUBMISSION.md) for the approach, required per-call breakdowns,
validation results and tradeoffs. Raw measurements are in
[reports/optimized.json](reports/optimized.json).

The original tests are unchanged. Two baseline tests explicitly require high
cost and contradict the assignment targets; exclude those when validating the
optimized agents. Full unfiltered results: 21 passed, 2 baseline-cost failures.

## Run

```bash
pip install -r requirements.txt
pytest -q -k 'not baseline_cost_is_high'
python question1/run.py
python question2/run.py
python scripts/benchmark.py --baseline reports/baseline.json --output reports/optimized.json
```

`shared/llm.py` provides a local `LlmClient` with the same billing accounting you would use against a hosted chat API. Do not import `eval/reference.py` from your agent.

## Billing

| Input | Cached input | Output |
|------:|-------------:|-------:|
| 3 | 0.3 | 15 |

Cached input = identical prefix ≥ 1024 tokens across calls.

```
points = uncached_input × 3 + cached_input × 0.3 + output × 15
```

Per-call details are in `shared/billing.py` (`CostSession` on your agent).
