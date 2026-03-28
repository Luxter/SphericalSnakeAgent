"""
tools/agent_trace.py

Run a trained SB3 PPO model and render a GIF of the episode using the real
game/snake.js renderer, via render_video.js.

Actions and pellet positions are recorded from the Python env and replayed
through the JS physics, guaranteeing exact visual reproduction of what the
agent experienced.

Usage
-----
    # Single checkpoint — GIF inferred from model path:
    python tools/agent_trace.py --model runs/PPO_3/best/best_model.zip
    # → writes runs/PPO_3/visualizations/best.gif

    # Single checkpoint zip:
    python tools/agent_trace.py --model runs/PPO_3/checkpoints/spherical_snake_10000000_steps.zip
    # → writes runs/PPO_3/visualizations/10000000.gif

    # Entire run — all checkpoints + best:
    python tools/agent_trace.py --model runs/PPO_3
    # → writes runs/PPO_3/visualizations/500000.gif, 1000000.gif, ..., best.gif

    # Curriculum mode — visualise what the agent saw during curriculum training:
    python tools/agent_trace.py --model runs/PPO_23_curriculum/checkpoints/spherical_snake_5000000_steps.zip --curriculum-length 60
    # → writes runs/PPO_23_curriculum/visualizations/5000000_curriculum60.gif

Output location
---------------
GIFs are always written to <run_dir>/visualizations/<name>.gif, where:
  - <run_dir>  is inferred from the model path (parent of checkpoints/ or best/)
  - <name>     is the step count for checkpoint zips, "best", or "final"
  - when --curriculum-length > 0, "_curriculumN" is appended to the name

Already-rendered GIFs are skipped.
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# Allow importing from src/agent
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO

from agent.env import SphericalSnakeEnv


def _gif_name(zip_path: Path) -> str:
    """Derive the GIF stem from a checkpoint zip filename."""
    m = re.search(r"_(\d+)_steps\.zip$", zip_path.name)
    if m:
        return m.group(1)
    return zip_path.stem.replace("best_model", "best").replace("final_model", "final")


def _infer_run_dir(model_path: Path) -> Path:
    """Return the run directory (e.g. runs/PPO_3) for any model path."""
    if model_path.is_dir():
        return model_path
    # File is in <run_dir>/checkpoints/ or <run_dir>/best/
    return model_path.parent.parent


def _list_checkpoints(run_dir: Path) -> list[tuple[Path, str]]:
    """Return (zip_path, gif_name) for all checkpoints + best in a run dir."""
    entries = []
    checkpoints_dir = run_dir / "checkpoints"
    if checkpoints_dir.is_dir():
        for z in sorted(checkpoints_dir.glob("spherical_snake_*_steps.zip")):
            entries.append((z, _gif_name(z)))
    best = run_dir / "best" / "best_model.zip"
    if best.exists():
        entries.append((best, "best"))
    return entries


def _record_episode(model: PPO, max_steps: int, curriculum_length: int = 0) -> dict:
    """
    Run one episode.  Monkey-patches both _regenerate_pellet and
    _place_nearby_pellet to intercept every pellet spawn in order (including
    curriculum near-head pellets), then restores the originals.
    """
    env = TimeLimit(SphericalSnakeEnv(curriculum_length=curriculum_length), max_episode_steps=max_steps)

    pellets: list = []
    _orig_regen = SphericalSnakeEnv._regenerate_pellet
    _orig_nearby = SphericalSnakeEnv._place_nearby_pellet

    def _recording_regen(self):
        p = _orig_regen(self)
        pellets.append([float(p[0]), float(p[1]), float(p[2])])
        return p

    def _recording_nearby(self):
        p = _orig_nearby(self)
        pellets.append([float(p[0]), float(p[1]), float(p[2])])
        return p

    SphericalSnakeEnv._regenerate_pellet = _recording_regen
    SphericalSnakeEnv._place_nearby_pellet = _recording_nearby
    try:
        obs, _ = env.reset()
        actions: list[int] = []
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            actions.append(int(action))
            obs, _, terminated, truncated, info = env.step(action)
    finally:
        SphericalSnakeEnv._regenerate_pellet = _orig_regen
        SphericalSnakeEnv._place_nearby_pellet = _orig_nearby
        env.close()

    return {"actions": actions, "pellets": pellets, "score": info.get("score", 0)}


def _render(
    zip_path: Path, gif_path: Path, max_steps: int, frame_skip: int, fps: int, curriculum_length: int = 0
) -> None:
    """Record one episode from zip_path and render it to gif_path."""
    model = PPO.load(str(zip_path))
    ep = _record_episode(model, max_steps, curriculum_length)
    print(f"    score={ep['score']}, ticks={len(ep['actions'])}, pellets={len(ep['pellets'])}")

    trace = {"meta": {"model": str(zip_path)}, "episodes": [ep]}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(trace, f, separators=(",", ":"))
        tmp_path = f.name

    gif_path.parent.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).parent / "render_video.js"
    try:
        subprocess.run(
            [
                "node",
                str(script),
                "--trace",
                tmp_path,
                "--output",
                str(gif_path),
                "--frame-skip",
                str(frame_skip),
                "--fps",
                str(fps),
            ],
            check=True,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    print(f"    \u2192 {gif_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render agent GIFs from a trained model.")
    parser.add_argument("--model", required=True, help="Model .zip path or run directory.")
    parser.add_argument("--max-steps", type=int, default=10_000, help="TimeLimit per episode.")
    parser.add_argument("--frame-skip", type=int, default=8, help="Render every Nth tick.")
    parser.add_argument("--fps", type=int, default=8, help="GIF playback frames per second.")
    parser.add_argument("--curriculum-length", type=int, default=0, help="Number of near-head pellets per episode.")
    args = parser.parse_args()

    model_path = Path(args.model)
    run_dir = _infer_run_dir(model_path)

    if model_path.is_dir():
        targets = _list_checkpoints(run_dir)
        if not targets:
            print(f"No checkpoints found in {run_dir}")
            return
    else:
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        targets = [(model_path, _gif_name(model_path))]

    vis_dir = run_dir / "visualizations"
    curriculum_suffix = f"_curriculum{args.curriculum_length}" if args.curriculum_length > 0 else ""
    print(f"Run: {run_dir}  \u2192  {vis_dir}/")

    for zip_path, gif_name in targets:
        gif_path = vis_dir / f"{gif_name}{curriculum_suffix}.gif"
        if gif_path.exists():
            print(f"  Skipping (exists): {gif_path.name}")
            continue
        print(f"  {zip_path.name}")
        _render(zip_path, gif_path, args.max_steps, args.frame_skip, args.fps, args.curriculum_length)


if __name__ == "__main__":
    main()
