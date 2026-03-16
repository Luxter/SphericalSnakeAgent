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
// Front 6 (indices 0-5): ±10°, ±30°, ±50° — narrow 20° cones, high forward resolution.
// Rear  4 (indices 6-9): ±90°, ±150°       — wide  60° cones, coarse background awareness.
// Tiling: front covers ±60°; ±90° covers 60°→120°; ±150° covers 120°→180°. Perfect 360°.
const _WHISKER_OFFSETS = [
     Math.PI / 18,        // +10°
    -Math.PI / 18,        // -10°
     Math.PI / 6,         // +30°
    -Math.PI / 6,         // -30°
     5 * Math.PI / 18,    // +50°
    -5 * Math.PI / 18,    // -50°
     Math.PI / 2,         // +90°
    -Math.PI / 2,         // -90°
     5 * Math.PI / 6,     // +150°
    -5 * Math.PI / 6,     // -150°
];
// Per-whisker half-angles (radians) and pre-computed cosines for cone tests.
// Front 6: π/18 (10°) — 20° total.  Rear 4: π/6 (30°) — 60° total.
// NODE_ANGLE is declared in snake.js (var NODE_ANGLE = Math.PI / 60).
const _WHISKER_HALF_ANGLES = [
    Math.PI / 18,  // +10°
    Math.PI / 18,  // -10°
    Math.PI / 18,  // +30°
    Math.PI / 18,  // -30°
    Math.PI / 18,  // +50°
    Math.PI / 18,  // -50°
    Math.PI / 6,   // +90°
    Math.PI / 6,   // -90°
    Math.PI / 6,   // +150°
    Math.PI / 6,   // -150°
];
const _WHISKER_COS_HALF = _WHISKER_HALF_ANGLES.map(h => Math.cos(h));
const _MAX_DIST = Math.PI;

// ---------------------------------------------------------------------------
// Observation computation — JS port of features.py : compute_obs()
//
// snake   : game's `snake` array; snake[0] is the head {x, y, z}
// pellet  : game's `pellet` {x, y, z}
// dir     : game's `direction` (radians)
// ---------------------------------------------------------------------------
function _computeObs(snake, pellet, dir) {
    const obs = new Float32Array(17);

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

    // --- indices 3-10 : whiskers ---
    const nNodes = snake.length;
    for (let wi = 0; wi < _WHISKER_OFFSETS.length; wi++) {
        const alpha   = dir + _WHISKER_OFFSETS[wi];
        const rayX    = -Math.cos(alpha);
        const rayY    = -Math.sin(alpha);
        const cosHalf = _WHISKER_COS_HALF[wi];
        let minArc    = _MAX_DIST;

        for (let j = 1; j < nNodes; j++) {
            const bx = snake[j].x, by = snake[j].y, bz = snake[j].z;
            const n2d = Math.sqrt(bx * bx + by * by);
            if (n2d < 1e-9) continue;
            if ((bx * rayX + by * rayY) / n2d < cosHalf) continue;
            // Subtract 2*NODE_ANGLE to get surface-to-surface gap instead of
            // center-to-center distance (0 = bodies touching = actual collision).
            const arc = Math.max(0.0, Math.acos(Math.max(-1.0, Math.min(1.0, -bz))) - 2 * NODE_ANGLE);
            if (arc < minArc) minArc = arc;
        }

        obs[3 + wi] = 1.0 - minArc / _MAX_DIST;
    }

    // --- index 13 : head z (invariantly -1 under world-rotation scheme) ---
    obs[13] = snake[0].z;

    // --- indices 14-15 : sin/cos of direction ---
    obs[14] = sinD;
    obs[15] = cosD;

    // --- index 16 : snake length normalised ---
    obs[16] = nNodes / 50.0;

    return obs;
}

// ---------------------------------------------------------------------------
// Whisker visualisation
// ---------------------------------------------------------------------------

// Arc-distance at which to draw the far edge of each whisker cone.
const _WHISKER_DISPLAY_ARC = Math.PI / 4;

// Latest obs stored so snake.js render() can call drawWhiskers().
var _lastObs = null;

/**
 * Project a unit-sphere point to canvas coordinates.
 * The sphere point is given as (rayX, rayY) in the tangent plane at arc
 * distance `arc` from the head (head is always at (0,0,-1)).
 */
function _sphereToScreen(rayX, rayY, arc) {
    const s = Math.sin(arc);
    const c = Math.cos(arc);
    const ex = s * rayX;
    const ey = s * rayY;
    const pz = -c + 2;          // ez = -cos(arc), then +2 for perspective
    return {
        sx: -ex * focalLength / pz + centerX,
        sy: -ey * focalLength / pz + centerY,
    };
}

/**
 * Draw the whisker cones from the snake head onto the canvas.
 * Each cone is ±22.5° wide; two edge lines and a filled triangle represent it.
 * Must be called from render() after the world has been rotated so
 * that the head is at (0, 0, -1) — which is always the case here.
 *
 * @param {Float32Array} obs  — the 15-element observation vector
 */
function drawWhiskers(obs) {
    const L = _WHISKER_DISPLAY_ARC;

    // Head projects to canvas centre.
    const hx = centerX;
    const hy = centerY;

    for (let wi = 0; wi < _WHISKER_OFFSETS.length; wi++) {
        const alpha = direction + _WHISKER_OFFSETS[wi];
        const HALF  = _WHISKER_HALF_ANGLES[wi];
        const val   = obs[3 + wi];   // 0 = safe, 1 = imminent collision

        // Centre ray direction for label / dot placement.
        const rayX = -Math.cos(alpha);
        const rayY = -Math.sin(alpha);

        // Screen endpoints of the two cone edges (left = −HALF, right = +HALF).
        const leftAlpha  = alpha - HALF;
        const rightAlpha = alpha + HALF;
        const left  = _sphereToScreen(-Math.cos(leftAlpha),  -Math.sin(leftAlpha),  L);
        const right = _sphereToScreen(-Math.cos(rightAlpha), -Math.sin(rightAlpha), L);

        // Colour: green (safe) → yellow → red (danger).
        const r = Math.min(255, Math.round(val * 2.0 * 255));
        const g = Math.min(255, Math.round((1.0 - val) * 2.0 * 255));

        // Filled cone triangle.
        ctx.beginPath();
        ctx.moveTo(hx, hy);
        ctx.lineTo(left.sx,  left.sy);
        ctx.lineTo(right.sx, right.sy);
        ctx.closePath();
        ctx.fillStyle = `rgba(${r},${g},0,0.1)`;
        ctx.fill();

        // Cone edge lines.
        ctx.strokeStyle = `rgba(${r},${g},0,0.2)`;
        ctx.lineWidth   = 1.0;
        ctx.beginPath();
        ctx.moveTo(hx, hy);
        ctx.lineTo(left.sx,  left.sy);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(hx, hy);
        ctx.lineTo(right.sx, right.sy);
        ctx.stroke();

        // Value label along the centre ray at ~60 % of display arc.
        const lbl  = _sphereToScreen(rayX, rayY, L * 0.6);
        const label = val.toFixed(2);
        ctx.font = "bold 9px monospace";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.strokeStyle = "rgba(0,0,0,0.7)";
        ctx.lineWidth = 2.5;
        ctx.strokeText(label, lbl.sx, lbl.sy);
        ctx.fillStyle = `rgba(${r},${g},0,1)`;
        ctx.fillText(label, lbl.sx, lbl.sy);
    }

    ctx.lineWidth = 1;  // restore default
    ctx.textAlign = "start";
    ctx.textBaseline = "alphabetic";
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
    _lastObs      = obs;   // expose for whisker visualisation
    const tensor  = new ort.Tensor("float32", obs, [1, obs.length]);
    const results = await _session.run({ obs: tensor });
    const probs   = results["action_probs"].data;   // Float32Array length 3

    let action = 0;
    if (probs[1] > probs[action]) action = 1;
    if (probs[2] > probs[action]) action = 2;

    setLeft(action === 1);
    setRight(action === 2);
}
