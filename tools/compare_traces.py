"""
tools/compare_traces.py

Diffs two JSON traces produced by snake_trace.js and snake_trace.py.
Reports per-tick, per-field divergences. Pass threshold: max |delta| < 1e-9.

Usage:
    python tools/compare_traces.py js_trace.json py_trace.json
"""

import json
import math
import sys


def flat_floats(obj, prefix=""):
    """Recursively flatten a JSON object into {dotted.key: float} pairs."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flat_floats(v, f"{prefix}.{k}" if prefix else k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(flat_floats(v, f"{prefix}[{i}]"))
    elif obj is None:
        pass  # null posQueue entries — skip
    else:
        out[prefix] = float(obj)
    return out


THRESHOLD = 1e-9


def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_traces.py <js_trace.json> <py_trace.json>")
        sys.exit(1)

    js_path, py_path = sys.argv[1], sys.argv[2]

    with open(js_path) as f:
        js_trace = json.load(f)
    with open(py_path) as f:
        py_trace = json.load(f)

    if len(js_trace) != len(py_trace):
        print(f"FAIL  trace length mismatch: JS={len(js_trace)} PY={len(py_trace)}")
        sys.exit(1)

    max_delta = 0.0
    failures = []

    for js_tick, py_tick in zip(js_trace, py_trace):
        t = js_tick["tick"]
        js_flat = flat_floats({k: v for k, v in js_tick.items() if k != "event"})
        py_flat = flat_floats({k: v for k, v in py_tick.items() if k != "event"})

        all_keys = sorted(set(js_flat) | set(py_flat))
        for key in all_keys:
            if key == "tick":
                continue
            js_val = js_flat.get(key)
            py_val = py_flat.get(key)
            if js_val is None or py_val is None:
                failures.append(
                    f"  tick {t:4d}  {key:<40s}  MISSING in {'JS' if js_val is None else 'PY'}"
                )
                continue
            delta = abs(js_val - py_val)
            if delta > max_delta:
                max_delta = delta
            if delta > THRESHOLD:
                failures.append(
                    f"  tick {t:4d}  {key:<40s}  JS={js_val:.15g}  PY={py_val:.15g}  Δ={delta:.3e}"
                )

        # Check event tags agree
        js_event = js_tick.get("event", "")
        py_event = py_tick.get("event", "")
        if js_event != py_event:
            failures.append(
                f"  tick {t:4d}  event mismatch: JS={js_event!r}  PY={py_event!r}"
            )

    if failures:
        print(f"FAIL  {len(failures)} divergence(s) found (max Δ={max_delta:.3e}):")
        for line in failures:
            print(line)
        sys.exit(1)
    else:
        print(
            f"PASS  All {len(js_trace)} ticks match."
            f"  Max delta: {max_delta:.9f}"
        )


if __name__ == "__main__":
    main()
