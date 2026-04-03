"""
ONNX export for the trained SphericalSnake PPO actor (Step 4).

Extracts the actor sub-network (mlp_extractor.policy_net → action_net →
softmax) from a stable-baselines3 PPO checkpoint and exports it to ONNX,
then writes game/agent_model.js with the model embedded as base64 so it can
be loaded from a file:// origin without a web server.

Usage
-----
    python -m agent.export_onnx runs/PPO_N/best/best_model.zip
    # → writes runs/PPO_N/best/best_model.onnx
    # → writes runs/PPO_N/best/best_model.js  (var AGENT_ONNX_B64_BEST_MODEL)
"""

import base64
from pathlib import Path

import onnx
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

    def forward(self, obs: torch.Tensor) -> torch.Tensor:  # (B, 21) → (B, 3)
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
    dummy = torch.zeros(1, 21, dtype=torch.float32)

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

    # Inline any external-data sidecar into a single self-contained file.
    model_proto = onnx.load(out_path, load_external_data=True)
    sidecar = Path(out_path + ".data")
    if sidecar.exists():
        sidecar.unlink()
    onnx.save(model_proto, out_path, save_as_external_data=False)
    print(f"Exported ONNX : {out_path}")

    # Write <checkpoint_name>.js next to the .onnx — base64-embedded for file:// origin.
    onnx_path = Path(out_path)
    js_name = onnx_path.stem
    js_var = "AGENT_ONNX_B64_" + js_name.upper().replace("-", "_")
    js_path = onnx_path.with_suffix(".js")
    b64 = base64.b64encode(onnx_path.read_bytes()).decode()
    js_path.write_text(f'var {js_var} = "{b64}";\n')
    print(f"Embedded JS   : {js_path}")


if __name__ == "__main__":
    app()
