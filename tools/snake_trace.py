"""
tools/snake_trace.py

Python counterpart to tools/snake_trace.js.
Runs the SphericalSnakeEnv physics with the same deterministic LCG RNG and
outputs an identical JSON trace for diffing.

Usage:
    python tools/snake_trace.py --seed 42 --actions '[1,1,1,0,2,1]'

The LCG state machine is identical to snake_trace.js:
    state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
    return state / 2**32
"""

import argparse
import json
import math
import sys
from pathlib import Path

# Allow importing from src/agent
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from agent.env import (
    SphericalSnakeEnv,
    NODE_ANGLE,
    NODE_QUEUE_SIZE,
    STARTING_DIRECTION,
    SNAKE_VELOCITY,
    TURN_RATE,
    INITIAL_SNAKE_LENGTH,
)


# ---------------------------------------------------------------------------
# Deterministic LCG — identical to snake_trace.js
# ---------------------------------------------------------------------------


class LCG:
    def __init__(self, seed: int) -> None:
        self.state = int(seed) & 0xFFFFFFFF

    def rand(self) -> float:
        self.state = (self.state * 1664525 + 1013904223) & 0xFFFFFFFF
        return self.state / 4294967296.0


# ---------------------------------------------------------------------------
# LCG-seeded subclass of SphericalSnakeEnv
# Overrides only _regenerate_pellet() to use the LCG.
# ---------------------------------------------------------------------------


class TracingEnv(SphericalSnakeEnv):
    def __init__(self, lcg: LCG) -> None:
        super().__init__()
        self._lcg = lcg

    def _regenerate_pellet(self) -> np.ndarray:
        """Mirrors JS regeneratePellet() using LCG instead of np_random."""
        theta = self._lcg.rand() * 2.0 * math.pi
        phi = self._lcg.rand() * math.pi
        sin_phi = math.sin(phi)
        return np.array(
            [math.cos(theta) * sin_phi, math.sin(theta) * sin_phi, math.cos(phi)],
            dtype=np.float64,
        )


def lcg_init(env: TracingEnv) -> None:
    """
    Replicate JS init() without going through gymnasium reset():
      1. Build pellet via LCG
      2. Build 8 snake nodes
    Mirrors the exact order in game/snake.js init().
    """
    env.direction = STARTING_DIRECTION
    env.score = 0
    env._terminated = False

    # JS calls regeneratePellet() BEFORE building snake nodes
    env.pellet = env._regenerate_pellet()

    env.snake = np.empty((0, 3), dtype=np.float64)
    env.pos_queues = np.empty((0, NODE_QUEUE_SIZE, 3), dtype=np.float64)
    for _ in range(INITIAL_SNAKE_LENGTH):
        env._add_snake_node()


# ---------------------------------------------------------------------------
# Snapshot helper — mirrors snake_trace.js snapshot()
# ---------------------------------------------------------------------------


def snapshot(env: TracingEnv, tick: int) -> dict:
    head = env.snake[0]
    # pos_queues[0] is now a (NODE_QUEUE_SIZE, 3) ndarray — all entries are real floats.
    # JS has null for unfilled slots (first 9 ticks); we output the pre-filled fallback
    # values instead.  compare_traces.py skips JS-side nulls, so these extra entries
    # are flagged as "MISSING in JS" for ticks 0-8 only.
    pq0 = [{"x": float(p[0]), "y": float(p[1]), "z": float(p[2])} for p in env.pos_queues[0]]
    return {
        "tick": tick,
        "direction": float(env.direction),
        "head": {"x": float(head[0]), "y": float(head[1]), "z": float(head[2])},
        "snake_length": len(env.snake),
        "pellet": {
            "x": float(env.pellet[0]),
            "y": float(env.pellet[1]),
            "z": float(env.pellet[2]),
        },
        "posQueue_0": pq0,
    }


# ---------------------------------------------------------------------------
# CLI harness
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--actions", type=str, default="[]")
    args = parser.parse_args()

    actions = json.loads(args.actions)
    lcg = LCG(args.seed)
    env = TracingEnv(lcg)
    lcg_init(env)

    trace = []
    for t, action in enumerate(actions):
        # --- Tick order mirrors JS update() while-loop exactly ---
        eaten, self_collision = env._check_collisions()

        if self_collision:
            snap = snapshot(env, t)
            snap["event"] = "terminated"
            trace.append(snap)
            break

        if action == 1:
            env.direction -= TURN_RATE
        elif action == 2:
            env.direction += TURN_RATE

        env._apply_snake_rotation()
        env._world_rotation()

        snap = snapshot(env, t)
        if eaten:
            snap["event"] = "pellet_eaten"
        trace.append(snap)

    print(json.dumps(trace, indent=2))


if __name__ == "__main__":
    main()
