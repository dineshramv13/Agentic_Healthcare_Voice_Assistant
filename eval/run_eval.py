"""
eval/run_eval.py

Eval runner: runs every query in eval/golden_set.json through the full
agent pipeline, scores each turn with RAGAS-style metrics + LLM-as-judge,
tracks latency, and writes a full report plus a console regression summary.

Also supports A/B testing system_prompt v1 vs v2 (--prompt-version flag) —
this is the practical mechanic behind the "prompts/registry.yaml is
A/B-test ready" claim from Phase 2: run the same golden set against both
versions and compare faithfulness/relevance scores directly.

Usage:
    python eval/run_eval.py
    python eval/run_eval.py --prompt-version v2
    python eval/run_eval.py --limit 10          # quick smoke test, first 10 queries only
"""

import sys
import os
import json
import time
import argparse
import logging
from datetime import datetime, timezone
from typing import Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.settings import settings
from agent.graph import build_graph
from agent.nodes.generator import GeneratorNode
from eval.metrics import RAGASMetrics, LLMJudge

logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

GOLDEN_SET_PATH = os.path.join(os.path.dirname(__file__), "golden_set.json")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "eval_results")


def load_golden_set() -> List[Dict]:
    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def check_themes_present(response: str, expected_themes: List[str]) -> float:
    """
    Simple, fast, non-LLM sanity check: fraction of expected_themes whose
    keywords appear (case-insensitively) anywhere in the response. This is
    a cheap complement to the LLM-based answer_relevance score — cheap
    enough to run on every turn with zero extra LLM calls, and useful for
    catching obviously broken responses (e.g. totally empty output).
    """
    if not expected_themes:
        return 1.0
    response_lower = response.lower()
    hits = sum(1 for theme in expected_themes if theme.lower() in response_lower)
    return hits / len(expected_themes)


def run_single_query(graph, query_entry: Dict, prompt_version: str) -> Dict:
    """Runs one golden set query through the full graph and returns the raw result + timing."""
    start = time.perf_counter()

    result = graph.invoke({
        "user_message": query_entry["query"],
        "session_id": f"eval-{query_entry['id']}",
        "trace_id": f"eval-trace-{query_entry['id']}",
        "conversation_history": "(no prior turns)",
        "retry_count": 0,
    })

    latency_ms = (time.perf_counter() - start) * 1000

    return {
        "id": query_entry["id"],
        "query": query_entry["query"],
        "expected_intent": query_entry["expected_intent"],
        "actual_intent": result.get("intent"),
        "intent_correct": result.get("intent") == query_entry["expected_intent"],
        "is_emergency_expected": query_entry["is_emergency"],
        "response": result.get("final_response") or result.get("response") or "",
        "retrieved_chunks": result.get("retrieved_chunks", []),
        "verified": result.get("verified"),
        "retry_count": result.get("retry_count", 0),
        "latency_ms": round(latency_ms, 2),
        "theme_coverage": check_themes_present(
            result.get("final_response") or result.get("response") or "",
            query_entry["expected_themes"],
        ),
    }


def run_eval(prompt_version: str = "v1", limit: int | None = None) -> Dict:
    golden_set = load_golden_set()
    if limit:
        golden_set = golden_set[:limit]

    logger.info("Running eval on %d queries with system_prompt version '%s'", len(golden_set), prompt_version)

    # Build a graph with the requested prompt version wired into the generator —
    # this is what makes A/B testing v1 vs v2 a one-flag operation rather than
    # a code change.
    generator_node = GeneratorNode(prompt_version=prompt_version)
    graph = build_graph(generator_node=generator_node)

    ragas = RAGASMetrics()
    judge = LLMJudge()

    turn_results = []
    for entry in golden_set:
        logger.info("Evaluating query %d/%d: '%s'", entry["id"], len(golden_set), entry["query"][:60])
        turn = run_single_query(graph, entry, prompt_version)

        # Skip the (expensive, 3-LLM-call) RAGAS scoring for emergency/out_of_scope
        # turns — those are hardcoded, non-LLM-generated responses by design
        # (see agent/nodes/emergency.py and fallback.py), so "faithfulness to
        # retrieved context" isn't a meaningful question for them.
        if turn["actual_intent"] in ("emergency", "out_of_scope") or not turn["retrieved_chunks"]:
            turn["ragas"] = {"faithfulness": None, "answer_relevance": None, "context_precision": None}
            turn["llm_judge_score"] = judge.score(entry["query"], turn["response"])
        else:
            turn["ragas"] = ragas.score_turn(entry["query"], turn["retrieved_chunks"], turn["response"])
            turn["llm_judge_score"] = judge.score(entry["query"], turn["response"])

        turn_results.append(turn)

    return _build_report(turn_results, prompt_version)


def _safe_mean(values: List[float]) -> float | None:
    clean = [v for v in values if v is not None]
    return round(sum(clean) / len(clean), 4) if clean else None


def _percentile(values: List[float], pct: float) -> float | None:
    """Returns the pct-th percentile (e.g. pct=95) of a list of numbers. Explicitly clamps the index so it never goes out of bounds, regardless of list size."""
    if not values:
        return None
    sorted_values = sorted(values)
    index = max(0, min(len(sorted_values) - 1, int(round(len(sorted_values) * (pct / 100))) - 1))
    return round(sorted_values[index], 2)


def _build_report(turn_results: List[Dict], prompt_version: str) -> Dict:
    intent_accuracy = sum(1 for t in turn_results if t["intent_correct"]) / len(turn_results)

    emergency_turns = [t for t in turn_results if t["is_emergency_expected"]]
    emergency_recall = (
        sum(1 for t in emergency_turns if t["actual_intent"] == "emergency") / len(emergency_turns)
        if emergency_turns else None
    )

    latencies = [t["latency_ms"] for t in turn_results]
    faithfulness_scores = [t["ragas"]["faithfulness"] for t in turn_results]
    relevance_scores = [t["ragas"]["answer_relevance"] for t in turn_results]
    precision_scores = [t["ragas"]["context_precision"] for t in turn_results]
    judge_scores = [t["llm_judge_score"] for t in turn_results]

    report = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "prompt_version": prompt_version,
        "num_queries": len(turn_results),
        "summary": {
            "intent_accuracy": round(intent_accuracy, 4),
            "emergency_recall": round(emergency_recall, 4) if emergency_recall is not None else None,
            "mean_faithfulness": _safe_mean(faithfulness_scores),
            "mean_answer_relevance": _safe_mean(relevance_scores),
            "mean_context_precision": _safe_mean(precision_scores),
            "mean_llm_judge_score": _safe_mean(judge_scores),
            "mean_theme_coverage": _safe_mean([t["theme_coverage"] for t in turn_results]),
            "mean_latency_ms": _safe_mean(latencies),
            "p95_latency_ms": _percentile(latencies, 95),
            "verification_failure_rate": round(
                sum(1 for t in turn_results if t["verified"] is False) / len(turn_results), 4
            ),
            "mean_retry_count": _safe_mean([t["retry_count"] for t in turn_results]),
        },
        "turns": turn_results,
    }
    return report


def print_regression_summary(report: Dict):
    s = report["summary"]
    print("\n" + "=" * 60)
    print(f"EVAL REPORT — system_prompt version: {report['prompt_version']}")
    print(f"Queries evaluated: {report['num_queries']}")
    print("=" * 60)
    print(f"Intent classification accuracy:  {s['intent_accuracy']:.1%}")
    if s["emergency_recall"] is not None:
        print(f"Emergency recall (caught all?):  {s['emergency_recall']:.1%}")
    print(f"Mean faithfulness:               {s['mean_faithfulness']}")
    print(f"Mean answer relevance:           {s['mean_answer_relevance']}")
    print(f"Mean context precision:          {s['mean_context_precision']}")
    print(f"Mean LLM-judge score (1-5):      {s['mean_llm_judge_score']}")
    print(f"Mean theme coverage:             {s['mean_theme_coverage']}")
    print(f"Mean latency:                    {s['mean_latency_ms']} ms")
    print(f"P95 latency:                     {s['p95_latency_ms']} ms")
    print(f"Verification failure rate:       {s['verification_failure_rate']:.1%}")
    print(f"Mean retry count:                {s['mean_retry_count']}")
    print("=" * 60)


def save_report(report: Dict) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    filename = f"report_{date_str}_{report['prompt_version']}.json"
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    return path


def main():
    parser = argparse.ArgumentParser(description="Run AI-Local's evaluation harness against the golden set")
    parser.add_argument("--prompt-version", default="v1", choices=["v1", "v2"], help="Which system_prompt version to evaluate (default: v1)")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N queries (for a quick smoke test)")
    args = parser.parse_args()

    report = run_eval(prompt_version=args.prompt_version, limit=args.limit)
    print_regression_summary(report)

    output_path = save_report(report)
    print(f"\nFull report saved to: {output_path}")


if __name__ == "__main__":
    main()
