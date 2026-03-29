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
// 18 whiskers at 20° spacing, uniform 20° cones (10° half-angle each).
// One exactly at 0° (front) and one at 180° (back); symmetric ± pairs in between.
// Coverage: 18 × 20° = 360° with zero gaps and zero overlaps.
const _WHISKER_OFFSETS = [
     0,                    //   0° front
     Math.PI / 9,          // +20°
    -Math.PI / 9,          // -20°
     2 * Math.PI / 9,      // +40°
    -2 * Math.PI / 9,      // -40°
     Math.PI / 3,          // +60°
    -Math.PI / 3,          // -60°
     4 * Math.PI / 9,      // +80°
    -4 * Math.PI / 9,      // -80°
     5 * Math.PI / 9,      // +100°
    -5 * Math.PI / 9,      // -100°
     2 * Math.PI / 3,      // +120°
    -2 * Math.PI / 3,      // -120°
     7 * Math.PI / 9,      // +140°
    -7 * Math.PI / 9,      // -140°
     8 * Math.PI / 9,      // +160°
    -8 * Math.PI / 9,      // -160°
     Math.PI,              // 180° back
];
// All whiskers use the same 10° half-angle — uniform 20° total cone.
// NODE_ANGLE is declared in snake.js (var NODE_ANGLE = Math.PI / 60).
const _WHISKER_HALF_ANGLES = _WHISKER_OFFSETS.map(() => Math.PI / 18);
const _WHISKER_COS_HALF = _WHISKER_HALF_ANGLES.map(h => Math.cos(h));
// NOTE: NODE_ANGLE is declared in snake.js — must be loaded before agent.js.
const _SIN_NODE_ANGLE = Math.sin(NODE_ANGLE);  // apparent-width cone expansion
const _MAX_DIST = Math.PI;

// ---------------------------------------------------------------------------
// Observation computation — JS port of features.py : compute_obs()
//
// snake   : game's `snake` array; snake[0] is the head {x, y, z}
// pellet  : game's `pellet` {x, y, z}
// dir     : game's `direction` (radians)
// ---------------------------------------------------------------------------
function _computeObs(snake, pellet, dir) {
    const obs = new Float32Array(21);

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
        const alpha     = dir + _WHISKER_OFFSETS[wi];
        const rayX      = -Math.cos(alpha);
        const rayY      = -Math.sin(alpha);
        const halfAngle = _WHISKER_HALF_ANGLES[wi];
        let minArc      = _MAX_DIST;

        for (let j = 2; j < nNodes; j++) {
            const bx = snake[j].x, by = snake[j].y, bz = snake[j].z;
            const n2d = Math.sqrt(bx * bx + by * by);
            if (n2d < 1e-9) continue;

            const cosLat = (bx * rayX + by * rayY) / n2d;

            // Dynamic apparent-width cone expansion: at close range a segment's
            // angular half-size asin(sin(NODE_ANGLE)/sin(d)) >> NODE_ANGLE, so
            // a fixed threshold misses segments whose center is just outside the
            // nominal cone while their body fully blocks the path.
            const arcToCenter = Math.acos(Math.max(-1.0, Math.min(1.0, -bz)));

            // Apparent-width expansion: only within the near hemisphere (arc <= PI/2).
            // Beyond that, a segment cannot physically overlap a cone edge any more
            // than at nominal size, and the formula diverges near the antipodal point.
            let cosEffective;
            if (arcToCenter <= Math.PI / 2) {
                const sinArc = Math.sin(arcToCenter);
                if (sinArc < _SIN_NODE_ANGLE) {
                    cosEffective = -1.0;  // segment on the head: hits all cones
                } else {
                    const apparentHw = Math.asin(_SIN_NODE_ANGLE / sinArc);
                    cosEffective = Math.cos(halfAngle + apparentHw);
                }
            } else {
                cosEffective = Math.cos(halfAngle);  // original nominal threshold
            }
            if (cosLat < cosEffective) continue;  // body does not reach into cone

            // Projected arc: arcToCenter * cosLat gives forward-direction
            // clearance; subtract 2*NODE_ANGLE for surface-to-surface gap.
            const arc = Math.max(0.0, arcToCenter * cosLat - 2 * NODE_ANGLE);
            if (arc < minArc) minArc = arc;
        }

        obs[3 + wi] = 1.0 - minArc / _MAX_DIST;
    }

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
        ctx.fillStyle = `rgba(${r},${g},0,0.05)`;
        ctx.fill();

        // Cone edge lines.
        ctx.strokeStyle = `rgba(${r},${g},0,0.12)`;
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
