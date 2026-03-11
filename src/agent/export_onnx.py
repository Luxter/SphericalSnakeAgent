"""
ONNX export for the trained SphericalSnake PPO actor (Step 4).

Extracts the actor sub-network (mlp_extractor.policy_net → action_net →
softmax) from a stable-baselines3 PPO checkpoint and exports it to ONNX.

Usage
-----
    python -m agent.export_onnx runs/PPO_9/best/best_model.zip
    # → writes runs/PPO_9/best/best_model.onnx
"""

from pathlib import Path

import torch
import torch.nn as nn
import typer
from stable_baselines3 import PPO


class _Actor(nn.Module):
    """Thin wrapper that chains policy_net → action_net → softmax."""

    def __init__(self, policy):
        super().__init__()
        self.policy_net = policy.mlp_extractor.policy_net
        self.action_net = policy.action_net

    def forward(self, obs: torch.Tensor) -> torch.Tensor:  # (B, 15) → (B, 3)
        latent = self.policy_net(obs)
        logits = self.action_net(latent)
        return torch.softmax(logits, dim=-1)


app = typer.Typer()


@app.command()
def main(
    model_path: str = typer.Argument(help="Path to .zip checkpoint."),
):
    print(f"Loading model : {model_path}")
    ppo = PPO.load(model_path, device="cpu")
    policy = ppo.policy
    policy.eval()

    actor = _Actor(policy)
    actor.eval()

    out_path = str(Path(model_path).with_suffix(".onnx"))
    dummy = torch.zeros(1, 15, dtype=torch.float32)

    batch = torch.export.Dim("batch", min=1, max=1024)
    torch.onnx.export(
        actor,
        dummy,
        out_path,
        input_names=["obs"],
        output_names=["action_probs"],
        dynamic_shapes={"obs": {0: batch}},
        opset_version=18,
    )
    print(f"Exported ONNX : {out_path}")


if __name__ == "__main__":
    app()
