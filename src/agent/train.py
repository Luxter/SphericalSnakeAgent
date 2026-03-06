"""
Training script for SphericalSnakeAgent (Step 3).

Usage
-----
From the repo root (or inside the dev container):

    python -m agent.train                   # train with defaults
    python -m agent.train --timesteps 500000
    python -m agent.train --timesteps 200000000 --n-envs 16 --run-dir runs/
    python -m agent.train --help            # show all options

After training, all artefacts for run N live under runs/PPO_N/:
  runs/PPO_N/checkpoints/   - periodic weight snapshots
  runs/PPO_N/best/          - best model found during evaluation
  runs/PPO_N/               - TensorBoard event file

    tensorboard --logdir runs/
"""

import os

import typer
from agent.env import SphericalSnakeEnv
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

# Repo root = two levels above this file (src/agent/train.py → src/ → repo root)
_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _next_run_id(run_dir: str, prefix: str = "PPO") -> int:
    """Return the next run index that SB3 would assign for *prefix*."""
    if not os.path.isdir(run_dir):
        return 1
    indices = []
    for name in os.listdir(run_dir):
        if name.startswith(f"{prefix}_") and os.path.isdir(os.path.join(run_dir, name)):
            try:
                indices.append(int(name[len(prefix) + 1 :]))
            except ValueError:
                pass
    return max(indices, default=0) + 1


def make_env(max_episode_steps: int = 10_000):
    """Factory used by make_vec_env / SubprocVecEnv."""
    return TimeLimit(SphericalSnakeEnv(), max_episode_steps=max_episode_steps)


app = typer.Typer()


@app.command()
def main(
    timesteps: int = typer.Option(200_000_000, help="Total environment steps."),
    n_envs: int = typer.Option(16, help="Number of parallel environments."),
    checkpoint_freq: int = typer.Option(500_000, help="Save a checkpoint every N *total* steps."),
    eval_freq: int = typer.Option(250_000, help="Run evaluation every N *total* steps."),
    eval_episodes: int = typer.Option(10, help="Episodes per evaluation."),
    max_episode_steps: int = typer.Option(10_000, help="TimeLimit per episode in steps."),
):
    run_dir = os.path.join(_REPO_ROOT, "runs")
    run_id = _next_run_id(run_dir)
    run_folder = os.path.join(run_dir, f"PPO_{run_id}")
    checkpoint_dir = os.path.join(run_folder, "checkpoints")
    best_model_dir = os.path.join(run_folder, "best")

    train_env = make_vec_env(
        make_env,
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={"max_episode_steps": max_episode_steps},
    )

    eval_env = make_vec_env(
        make_env,
        n_envs=1,
        env_kwargs={"max_episode_steps": max_episode_steps},
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        n_steps=2048,
        batch_size=512,
        n_epochs=10,
        learning_rate=3e-4,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=run_dir,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(checkpoint_freq // n_envs, 1),
        save_path=checkpoint_dir,
        name_prefix="spherical_snake",
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=best_model_dir,
        log_path=run_folder,
        eval_freq=max(eval_freq // n_envs, 1),
        n_eval_episodes=eval_episodes,
        deterministic=True,
        verbose=1,
    )

    print(
        f"Training for {timesteps:,} steps "
        f"across {n_envs} parallel envs.\n"
        f"Run folder:  {run_folder}\n"
        f"TensorBoard: tensorboard --logdir {run_dir}"
    )

    model.learn(
        total_timesteps=timesteps,
        callback=[checkpoint_cb, eval_cb],
        tb_log_name="PPO",
        progress_bar=True,
    )

    final_path = os.path.join(run_folder, "final_model")
    model.save(final_path)
    print(f"\nFinal model saved to {final_path}.zip")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    app()
