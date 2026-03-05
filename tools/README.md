# tools — Physics Parity Verification

These scripts verify that `src/agent/env.py` produces bit-for-bit identical state
to `game/snake.js` by running both with the same deterministic inputs and diffing
the output.

## Files

| File | Purpose |
|---|---|
| `snake_trace.js` | Loads `game/snake.js` unmodified via Node's `vm.runInThisContext` with minimal DOM stubs. Replaces `Math.random` with a deterministic LCG. Accepts `--seed` and `--actions`, writes a JSON trace to stdout. |
| `snake_trace.py` | Python counterpart. Drives `SphericalSnakeEnv` physics with the same LCG. Accepts identical CLI args, writes identical JSON structure. |
| `compare_traces.py` | Diffs two trace files field-by-field. Reports any divergence with tick, field name, JS value, Python value, and delta. Pass threshold: `max \|delta\| < 1e-9`. |

## Trace record format

Each tick produces one JSON object:

```json
{
  "tick": 42,
  "direction": 0.785398,
  "head": { "x": 0.0, "y": 0.0, "z": -1.0 },
  "snake_length": 8,
  "pellet": { "x": 0.312, "y": -0.756, "z": 0.575 },
  "posQueue_0": [ { "x": ..., "y": ..., "z": ... }, null, ... ],
  "event": "pellet_eaten"   // optional: "pellet_eaten" | "terminated"
}
```

## Running the checks

### Straight-line (100 ticks, all action=1)
Validates rotation math and posQueue propagation.

```bash
ACTIONS=$(python3 -c "import json; print(json.dumps([0]*100))")
node tools/snake_trace.js --seed 42 --actions "$ACTIONS" > /tmp/js_trace.json
python3 tools/snake_trace.py --seed 42 --actions "$ACTIONS" > /tmp/py_trace.json
python3 tools/compare_traces.py /tmp/js_trace.json /tmp/py_trace.json
# Expected: PASS  All 100 ticks match.  Max delta: 0.000000000
```

### Pellet-eat (400 ticks, seed 8 eats at tick 291)
Validates collision detection, `addSnakeNode`, and queue continuity across a length change.

```bash
ACTIONS=$(python3 -c "import json; print(json.dumps([0]*400))")
node tools/snake_trace.js --seed 8 --actions "$ACTIONS" > /tmp/js_eat.json
python3 tools/snake_trace.py --seed 8 --actions "$ACTIONS" > /tmp/py_eat.json
python3 tools/compare_traces.py /tmp/js_eat.json /tmp/py_eat.json
# Expected: PASS  All 400 ticks match.  Max delta: 0.000000000
```

## How it works

`snake_trace.js` sets up DOM stubs (`document.querySelector`, `window.addEventListener`,
etc.) then executes `game/snake.js` verbatim using `vm.runInThisContext`. This brings
all of snake.js's globals (`snake`, `pellet`, `direction`, `applySnakeRotation`, …)
into scope with zero code duplication. `Math.random` is patched to the LCG **before**
the script loads so `init() → regeneratePellet()` uses it from the first call.

`snake_trace.py` subclasses `SphericalSnakeEnv` and overrides only `_regenerate_pellet()`
to use the same LCG, ensuring identical pellet placement across both runtimes.

The LCG shared by both scripts:
```
state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
return state / 2**32
```

## Notes

- Grid points (`points[]` in snake.js) are included in the JS world rotation via
  `allPoints()` but are rendering-only. They are excluded from `env.py` and from
  the trace. This is intentional and does not affect any physics state.
- `posQueue` entries that are still `null` (first 9 ticks per node) are skipped
  in both JS and Python rotation, and appear as `null` in the trace.
