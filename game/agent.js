/**
 * agent.js — In-browser PPO inference for Spherical Snake (Step 5)
 *
 * Depends on (must be loaded first):
 *   ort.min.js      — onnxruntime-web
 *   agent_model.js  — defines AGENT_ONNX_B64 (base64-encoded agent.onnx)
 *
 * Exposes:
 *   agentReady  {boolean}  — true once the ONNX session is loaded
 *   agentStep() {void}     — call once per game tick in AI mode
 */

"use strict";

// ---------------------------------------------------------------------------
// Constants — must match features.py exactly
// ---------------------------------------------------------------------------
const _WHISKER_OFFSETS = [
     0.0,                //   0°    — front
     Math.PI / 8,        // +22.5°
    -Math.PI / 8,        // -22.5°
     3 * Math.PI / 8,    // +67.5°
    -3 * Math.PI / 8,    // -67.5°
     5 * Math.PI / 8,    // +112.5°
    -5 * Math.PI / 8,    // -112.5°
     7 * Math.PI / 8,    // +157.5°
    -7 * Math.PI / 8,    // -157.5°
];
const _COS_HALF = Math.cos(Math.PI / 8);   // cos(22.5°)
const _MAX_DIST = Math.PI;

// ---------------------------------------------------------------------------
// Observation computation — JS port of features.py : compute_obs()
//
// snake   : game's `snake` array; snake[0] is the head {x, y, z}
// pellet  : game's `pellet` {x, y, z}
// dir     : game's `direction` (radians)
// ---------------------------------------------------------------------------
function _computeObs(snake, pellet, dir) {
    const obs = new Float32Array(16);

    const cosD = Math.cos(dir);
    const sinD = Math.sin(dir);

    // Heading and right tangent vectors at south-pole head (0, 0, -1).
    const hx = -cosD, hy = -sinD;   // forward
    const rx =  sinD, ry = -cosD;   // right (90° CW)

    // --- indices 0-1 : pellet bearing (sin, cos) ---
    const px = pellet.x, py = pellet.y;
    const projNorm = Math.sqrt(px * px + py * py);
    if (projNorm < 1e-9) {
        obs[0] = 0.0;
        obs[1] = 1.0;
    } else {
        const ppx = px / projNorm, ppy = py / projNorm;
        obs[0] = ppx * rx + ppy * ry;   // right component → bearing_sin
        obs[1] = ppx * hx + ppy * hy;   // fwd  component → bearing_cos
    }

    // --- index 2 : pellet great-circle distance / π ---
    // head is always (0, 0, -1) so dot(head, pellet) = -pellet.z
    obs[2] = Math.acos(Math.max(-1.0, Math.min(1.0, -pellet.z))) / _MAX_DIST;

    // --- indices 3-11 : whiskers ---
    const nNodes = snake.length;
    for (let wi = 0; wi < _WHISKER_OFFSETS.length; wi++) {
        const alpha = dir + _WHISKER_OFFSETS[wi];
        const rayX  = -Math.cos(alpha);
        const rayY  = -Math.sin(alpha);
        let minArc  = _MAX_DIST;

        for (let j = 1; j < nNodes; j++) {
            const bx = snake[j].x, by = snake[j].y, bz = snake[j].z;
            const n2d = Math.sqrt(bx * bx + by * by);
            if (n2d < 1e-9) continue;
            if ((bx * rayX + by * rayY) / n2d < _COS_HALF) continue;
            const arc = Math.acos(Math.max(-1.0, Math.min(1.0, -bz)));
            if (arc < minArc) minArc = arc;
        }

        obs[3 + wi] = 1.0 - minArc / _MAX_DIST;
    }

    // --- index 12 : head z (invariantly -1 under world-rotation scheme) ---
    obs[12] = snake[0].z;

    // --- indices 13-14 : sin/cos of direction ---
    obs[13] = sinD;
    obs[14] = cosD;

    // --- index 15 : snake length normalised ---
    obs[15] = nNodes / 50.0;

    return obs;
}

// ---------------------------------------------------------------------------
// ONNX session
// ---------------------------------------------------------------------------
var agentReady = false;
var _session   = null;

(async function _load() {
    try {
        const raw   = atob(AGENT_ONNX_B64);
        const bytes = new Uint8Array(raw.length);
        for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
        _session   = await ort.InferenceSession.create(bytes);
        agentReady = true;
        console.log("[agent] ONNX model loaded.");
    } catch (e) {
        console.error("[agent] Failed to load ONNX model:", e);
    }
})();

// ---------------------------------------------------------------------------
// Per-tick inference
// ---------------------------------------------------------------------------

// action 0 = STRAIGHT, 1 = LEFT (direction -= 0.08), 2 = RIGHT (direction += 0.08)
async function agentStep() {
    if (!agentReady) return;

    const obs     = _computeObs(snake, pellet, direction);
    const tensor  = new ort.Tensor("float32", obs, [1, obs.length]);
    const results = await _session.run({ obs: tensor });
    const probs   = results["action_probs"].data;   // Float32Array length 3

    let action = 0;
    if (probs[1] > probs[action]) action = 1;
    if (probs[2] > probs[action]) action = 2;

    setLeft(action === 1);
    setRight(action === 2);
}
