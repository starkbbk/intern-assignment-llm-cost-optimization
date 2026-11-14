"""Reproduce uncached per-ticket costs using the unmodified billing/client."""
from __future__ import annotations

import argparse
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from question1.agent import SupportCopilot
from question2.agent import SupportPipeline

PUBLIC = ("T-1001", "T-1002", "T-1003", "T-1004")
HOLDOUT = "T-1091"
TRACKS = {
    "question1": (SupportCopilot, 3000),
    "question2": (SupportPipeline, 2500),
}


def benchmark() -> dict:
    report = {
        "runtime": {"python": platform.python_version(), "tiktoken": version("tiktoken")},
        "measurement": "Fresh agent and CostSession per ticket; no cached-prefix discount.",
        "tracks": {},
    }
    for name, (agent_type, target) in TRACKS.items():
        rows = []
        for ticket_id in (*PUBLIC, HOLDOUT):
            agent = agent_type()
            result = agent.handle_ticket(ticket_id)
            calls = [
                {
                    "label": call.label,
                    "input_tokens": call.input_tokens,
                    "uncached_input_tokens": call.input_tokens - call.cached_prefix_tokens,
                    "cached_input_tokens": call.cached_prefix_tokens,
                    "output_tokens": call.output_tokens,
                    "points": call.points(),
                }
                for call in agent.session.calls
            ]
            rows.append({
                "ticket_id": ticket_id,
                "action": result.action,
                "kb_article": result.kb_article,
                "points": agent.session.total_points(),
                "target": target if ticket_id in PUBLIC else None,
                "calls": calls,
            })
            assert calls, f"{name}/{ticket_id}: missing billed LLM call"
            assert all(c["cached_input_tokens"] == 0 for c in calls)
            if ticket_id in PUBLIC:
                assert rows[-1]["points"] <= target, f"{name}/{ticket_id}: over budget"
        report["tracks"][name] = rows
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Also save the JSON report here")
    parser.add_argument("--baseline", type=Path, help="Compare routing with a saved baseline report")
    args = parser.parse_args()
    report = benchmark()
    if args.baseline:
        baseline = json.loads(args.baseline.read_text())
        for name, rows in report["tracks"].items():
            previous = {row["ticket_id"]: row for row in baseline["tracks"][name]}
            for row in rows:
                before = previous[row["ticket_id"]]
                assert (row["action"], row["kb_article"]) == (before["action"], before["kb_article"])
                row["baseline_points"] = before["points"]
                row["reduction_percent"] = round(100 * (1 - row["points"] / before["points"]), 2)
        report["baseline_source_commit"] = baseline["source_commit"]
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
