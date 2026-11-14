# ShopAssist cost optimization

Both tracks preserve `action` and `kb_article` for all four public tickets and
T-1091 using the supplied local client. Every public ticket meets its target on a
fresh agent and billing session, with zero cached input tokens.

## Results

| Ticket | Q1 baseline | Q1 optimized (≤3000) | Q2 baseline | Q2 optimized (≤2500) |
|---|---:|---:|---:|---:|
| T-1001 | 8862 | 1905 | 73401 | 1515 |
| T-1002 | 8796 | 1782 | 73245 | 1284 |
| T-1003 | 8751 | 1743 | 72990 | 1245 |
| T-1004 | 8892 | 1704 | 73386 | 1314 |
| **Public total** | **35301** | **7134** | **293022** | **5358** |
| T-1091 (holdout; no specified budget) | 8760 | 1746 | 73551 | 1269 |

Public-total reduction: Q1 **79.79%**; Q2 **98.17%**.

Baseline values are one measured run of commit
`d9989e294615046519a1658ff76c4343f465fb0c`; the baseline includes random UUIDs,
so its exact token counts can vary between runs. The optimized measurements use
Python 3.12.14, tiktoken 0.14.0 and `cl100k_base`.

## Q1: one call per ticket

`question1/agent.py` still makes exactly one `LlmClient.chat` call per ticket and
uses the existing parser and resolution type. It supplies the complete small
help center, including the subscription policy, rather than only the four public
examples. Every order field is retained, including `subscription_age_days`.

The prompt requests only the JSON object and a short customer reply. Removing
verbose reasoning instructions cuts output from roughly 480 tokens to 36–46 on
the public tickets. This has the largest effect because output costs 15 points
per token. Removing redundant examples and compacting the order JSON also cuts
input. The random run UUID is removed, keeping the static system prefix stable.
The trusted snapshot precedes customer text, so the simulator sees the actual
order before any snapshot-like text in the thread.

Required per-call breakdown, T-1001:

| Call | Uncached input | Cached input | Output | Points |
|---|---:|---:|---:|---:|
| handle_ticket | 405 | 0 | 46 | **1905** |

`405 × 3 + 0 × 0.3 + 46 × 15 = 1905`.

## Q2: retain four stages, use the LLM where needed

The four named nodes still execute in order:

1. **Triage:** local routing by trusted product metadata selects a policy scope.
   Physical items retain both return and damage policies; a customization flag
   also includes the custom-item exception. Unknown types fall back to all
   canonical article IDs. This selects documents, not a final action.
2. **Retrieve:** `load_article` reads the actual selected policy files. It avoids
   the expanded filler corpus, arbitrary character slices, and model-generated
   copies of articles.
3. **Compose:** one billed LLM call applies those policies to the full order and
   thread, returning concise JSON.
4. **Audit:** local parsing checks the action vocabulary and that the cited
   article was retrieved. Invalid output raises an error.

This interprets “four-node pipeline” as four logical stages, without requiring
four model invocations; the specification gives no minimum LLM-call count.
The deterministic nodes each cost **0 LLM points**, not unreported model calls.
All decisions continue through the supplied `LlmClient`; the agent does not
import `eval/reference.py`, hardcode ticket answers, change billing, claim cache
hits, or reuse previous ticket resolutions.

Required per-call breakdown for a physical fulfillment path (T-1004):

| Node | Uncached input | Cached input | Output | Points |
|---|---:|---:|---:|---:|
| triage (local) | 0 | 0 | 0 | 0 |
| retrieve (local) | 0 | 0 | 0 | 0 |
| compose | 258 | 0 | 36 | **1314** |
| audit (local) | 0 | 0 | 0 | 0 |

`258 × 3 + 36 × 15 = 1314`.

Required per-call breakdown for a denial path (T-1002):

| Node | Uncached input | Cached input | Output | Points |
|---|---:|---:|---:|---:|
| triage (local) | 0 | 0 | 0 | 0 |
| retrieve (local) | 0 | 0 | 0 | 0 |
| compose | 233 | 0 | 39 | **1284** |
| audit (local) | 0 | 0 | 0 | 0 |

`233 × 3 + 39 × 15 = 1284`.

## Cache accounting

The targets are met without caching. These prompts are below the repository's
1024-token cache threshold, so stabilizing a prefix alone earns no discount.
No artificial padding or `cached_prefix` argument is used. The benchmark reads
actual calls recorded by the unchanged `CostSession`, including input, cached
input, output and points.

## Validation and reproduction

From the repository root, with Python 3.10 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q -k 'not baseline_cost_is_high'
python scripts/benchmark.py --baseline reports/baseline.json --output reports/optimized.json
python question1/run.py
python question2/run.py
```

The original tests are unchanged. Two original tests named
`test_baseline_cost_is_high` require Q1's public total to exceed 15000 and Q2's
first-ticket cost to exceed 70000. These assertions describe the unoptimized
baseline and contradict the optimization targets: Q1's target implies a public
total of at most 12000, and Q2 requires at most 2500 per public ticket.
Consequently, an unfiltered `pytest -q` is expected to fail **only those two
baseline-cost assertions**. The filtered command above keeps all original
correctness tests and the large-help-center check. That help-center check can
take about a minute because it builds the original expanded corpus.

`tests/test_optimization.py` adds fresh-session and reused-session budget checks,
correct routing for all available tickets, one billed call per ticket, retrieval
fallback and customization coverage, and rejection of invalid audit output.
`scripts/benchmark.py` independently checks each public budget and compares
routing with the measured baseline. Both reports include all five ticket IDs.
Final unfiltered result: **21 passed, 2 failed**; the only failures are the two
contradictory baseline-cost assertions above. See `reports/pytest-full.txt` for
the complete result.

## Scope and tradeoffs

These are measurements of the supplied simulator, not hosted-model accuracy,
latency or real currency spend. Its routing uses the order snapshot internally,
and its verbose-output behavior is triggered by words anywhere in the prompt;
it does not prove that a real model follows the selected policy. Prompt
changes would need fresh accuracy evaluation with a real provider.

The local audit checks structure and citation scope; it does not independently
validate semantic policy eligibility. Replacing the LLM audit trades away that
second model review. A production rollout should evaluate this tradeoff on
representative tickets and route ambiguous decisions to review. The product
scope table also needs maintenance when new policies are added. Unknown-type
fallback preserves recall for today's five articles but needs indexed retrieval
if the canonical policy set becomes large.
