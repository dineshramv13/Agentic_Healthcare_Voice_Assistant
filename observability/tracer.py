"""
observability/tracer.py

TraceLogger: mimics LangSmith's trace structure locally, as JSONL files —
one line per node execution, one file per day. No external service, no
API key, zero cost.

Input:  trace_id, node_name, input_data, output_data, latency_ms, extra metadata
Output: appends one JSON line to traces/trace_YYYY-MM-DD.jsonl

Used via the `@traced` decorator (wraps any node's __call__) so every node
gets tracing for free without repeating boilerplate in each node file.
"""

import os
import json
import time
import logging
import functools
from datetime import datetime, timezone
from typing import Dict, Any, Callable

from config.settings import settings

logger = logging.getLogger(__name__)


class TraceLogger:
    def __init__(self, trace_dir: str | None = None):
        self.trace_dir = trace_dir or settings.trace_dir
        os.makedirs(self.trace_dir, exist_ok=True)

    def _current_trace_file(self) -> str:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return os.path.join(self.trace_dir, f"trace_{date_str}.jsonl")

    def log(
        self,
        trace_id: str,
        node_name: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        latency_ms: float,
        session_id: str | None = None,
        extra: Dict[str, Any] | None = None,
    ) -> None:
        """Appends one trace record as a JSON line."""
        record = {
            "trace_id": trace_id,
            "session_id": session_id,
            "node": node_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "latency_ms": round(latency_ms, 2),
            "input": input_data,
            "output": output_data,
            "extra": extra or {},
        }
        try:
            with open(self._current_trace_file(), "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as e:
            # Tracing must never crash the actual request — log and move on.
            logger.error("Failed to write trace record: %s", e)

    def read_traces(self, n: int | None = None, trace_id: str | None = None) -> list:
        """
        Reads trace records from today's file (used by GET /traces in api/server.py).
        If trace_id is given, filters to just that trace.
        If n is given, returns only the last n records.
        """
        path = self._current_trace_file()
        if not os.path.exists(path):
            return []

        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if trace_id is None or record.get("trace_id") == trace_id:
                    records.append(record)

        if n is not None:
            records = records[-n:]
        return records


# Singleton instance, shared across all nodes
tracer = TraceLogger()


def _truncate_for_trace(value: Any, max_len: int = 500) -> Any:
    """Keeps trace files readable — full chunk text etc. gets truncated."""
    if isinstance(value, str) and len(value) > max_len:
        return value[:max_len] + f"... [truncated, {len(value)} chars total]"
    if isinstance(value, list):
        return [_truncate_for_trace(v, max_len) for v in value[:5]]  # cap list length too
    if isinstance(value, dict):
        return {k: _truncate_for_trace(v, max_len) for k, v in value.items()}
    return value


def traced(node_name: str):
    """
    Decorator for node __call__ methods. Wraps the call, times it, and logs
    a trace record with truncated input/output snapshots.

    Usage (see any file in agent/nodes/ after this phase):
        class SafetyNode:
            @traced("safety_check")
            def __call__(self, state: AgentState) -> dict:
                ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, state: Dict, *args, **kwargs) -> Dict:
            start = time.perf_counter()
            output = func(self, state, *args, **kwargs)
            latency_ms = (time.perf_counter() - start) * 1000

            # Only trace a small, relevant slice of state as "input" rather
            # than the entire (possibly large) state dict every time.
            input_snapshot = _truncate_for_trace(
                {
                    "user_message": state.get("user_message"),
                    "intent": state.get("intent"),
                    "retry_count": state.get("retry_count"),
                }
            )
            output_snapshot = _truncate_for_trace(output)

            tracer.log(
                trace_id=state.get("trace_id", "unknown"),
                node_name=node_name,
                input_data=input_snapshot,
                output_data=output_snapshot,
                latency_ms=latency_ms,
                session_id=state.get("session_id"),
            )
            return output

        return wrapper

    return decorator
