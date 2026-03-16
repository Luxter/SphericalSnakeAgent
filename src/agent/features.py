"""
Feature extraction for SphericalSnakeEnv.

    compute_obs(snake, pellet, direction) -> np.ndarray[float32, (17,)]

Observation layout:
  Index | Feature
  ------|---------------------------------------------------------
  0     | pellet_bearing_sin  — sin of signed lateral angle head→pellet
  1     | pellet_bearing_cos  — cos of same (avoids ±π discontinuity)
  2     | pellet_dist         — great-circle distance head→pellet / π
  3-12  | whiskers[10]        — 10 rays; 0=safe, 1=imminent collision
            Front 6 (±10°, ±30°, ±50°): narrow 20° cones — high angular resolution
            Rear  4 (±90°, ±150°):       wide  60° cones — coarse background awareness
  13    | head_z              — z coordinate of head
  14    | sin(direction)
  15    | cos(direction)
  16    | snake_len_norm       — len(snake) / 50
"""

import math
import numpy as np

NODE_ANGLE: float = math.pi / 60  # angular radius of one snake node

# Whisker ray offsets from heading angle, indices 3-12.
# Front 6 (indices 0-5): ±10°, ±30°, ±50° — narrow cones, high forward resolution.
# Rear  4 (indices 6-9): ±90°, ±150°      — wide cones, coarse background awareness.
# Tiling: front covers ±60° (20° cones, no gaps/overlaps); rear picks up at ±60°,
# ±90° covers 60°→120°, ±150° covers 120°→180° — perfect 360° with zero overlap.
_WHISKER_OFFSETS: tuple = (
    math.pi / 18,  # +10°  (index 3)
    -math.pi / 18,  # -10°  (index 4)
    math.pi / 6,  # +30°  (index 5)
    -math.pi / 6,  # -30°  (index 6)
    5 * math.pi / 18,  # +50°  (index 7)
    -5 * math.pi / 18,  # -50°  (index 8)
    math.pi / 2,  # +90°  (index 9)
    -math.pi / 2,  # -90°  (index 10)
    5 * math.pi / 6,  # +150° (index 11)
    -5 * math.pi / 6,  # -150° (index 12)
)

# Per-whisker half-angles.  A body node is inside cone wi when its alignment
# dot-product with the ray direction exceeds _WHISKER_COS_HALF[wi].
# Front 6: π/18 (10°) — 20° total cone.  Rear 4: π/6 (30°) — 60° total cone.
# Each half-angle is expanded by NODE_ANGLE so that a node whose body overlaps
# a cone boundary is correctly detected even if its centre lies just outside.
_WHISKER_HALF_ANGLES: tuple = (
    math.pi / 18,  # +10°
    math.pi / 18,  # -10°
    math.pi / 18,  # +30°
    math.pi / 18,  # -30°
    math.pi / 18,  # +50°
    math.pi / 18,  # -50°
    math.pi / 6,  # +90°
    math.pi / 6,  # -90°
    math.pi / 6,  # +150°
    math.pi / 6,  # -150°
)
_WHISKER_COS_HALF: tuple = tuple(math.cos(h + NODE_ANGLE) for h in _WHISKER_HALF_ANGLES)

# Maximum great-circle distance used to normalise whisker and pellet readings.
_MAX_DIST: float = math.pi


def compute_obs(
    snake: np.ndarray,
    pellet: np.ndarray,
    direction: float,
) -> np.ndarray:
    """
    Return the 16-element float32 observation vector.

    Parameters
    ----------
    snake      : (N, 3) float64 — node positions; snake[0] is the head,
                 invariantly at (0, 0, -1) under the world-rotation scheme.
    pellet     : (3,) float64 — unit-sphere food position.
    direction  : float — current heading angle (radians).
    """
    obs = np.empty(17, dtype=np.float32)

    head = snake[0]
    cos_d = math.cos(direction)
    sin_d = math.sin(direction)

    # Heading and right tangent vectors (exact for head at south pole).
    h_x, h_y = -cos_d, -sin_d  # heading
    r_x, r_y = sin_d, -cos_d  # right (90° CW from heading)

    # ------------------------------------------------------------------
    # Indices 0-1 : pellet bearing (sin, cos)
    #
    # At head = (0,0,-1) the tangent-plane projection of pellet is (px, py, 0).
    # We decompose the unit projection direction into (forward, right) components.
    # ------------------------------------------------------------------
    px = float(pellet[0])
    py = float(pellet[1])
    proj_norm = math.sqrt(px * px + py * py)

    if proj_norm < 1e-9:
        # Pellet is at the head position — bearing undefined; default to ahead.
        obs[0] = 0.0  # bearing_sin
        obs[1] = 1.0  # bearing_cos
    else:
        ppx = px / proj_norm
        ppy = py / proj_norm
        obs[0] = float(ppx * r_x + ppy * r_y)  # right component  → bearing_sin
        obs[1] = float(ppx * h_x + ppy * h_y)  # fwd  component  → bearing_cos

    # ------------------------------------------------------------------
    # Index 2 : pellet great-circle distance / π
    # ------------------------------------------------------------------
    dot_hp = float(head[0] * pellet[0] + head[1] * pellet[1] + head[2] * pellet[2])
    obs[2] = math.acos(max(-1.0, min(1.0, dot_hp))) / _MAX_DIST

    # ------------------------------------------------------------------
    # Indices 3-11 : whiskers
    # ------------------------------------------------------------------
    n_nodes = len(snake)

    for wi, offset in enumerate(_WHISKER_OFFSETS):
        alpha = direction + offset
        ray_x = -math.cos(alpha)
        ray_y = -math.sin(alpha)

        min_arc = _MAX_DIST

        for j in range(1, n_nodes):
            # Tangent-plane projection at (0,0,-1) is just the xy components.
            bx = float(snake[j, 0])
            by = float(snake[j, 1])
            bz = float(snake[j, 2])

            n2d = math.sqrt(bx * bx + by * by)
            if n2d < 1e-9:
                continue  # node is at the south pole (on top of head)

            # Cosine of angle between node's projected direction and ray.
            cos_lat = (bx * ray_x + by * ray_y) / n2d
            if cos_lat < _WHISKER_COS_HALF[wi]:
                continue  # node is outside this whisker's cone

            # Great-circle arc distance: dot(b, head) = -bz for head=(0,0,-1).
            # Subtract 2*NODE_ANGLE to convert center-to-center distance to
            # surface-to-surface gap (0 = bodies touching, i.e. actual collision).
            arc = max(0.0, math.acos(max(-1.0, min(1.0, -bz))) - 2.0 * NODE_ANGLE)
            if arc < min_arc:
                min_arc = arc

        obs[3 + wi] = float(1.0 - min_arc / _MAX_DIST)

    # ------------------------------------------------------------------
    # Index 13 : head z coordinate
    # ------------------------------------------------------------------
    obs[13] = float(head[2])

    # ------------------------------------------------------------------
    # Indices 14-15 : sin/cos of direction
    # ------------------------------------------------------------------
    obs[14] = float(sin_d)
    obs[15] = float(cos_d)

    # ------------------------------------------------------------------
    # Index 16 : snake length normalised
    # ------------------------------------------------------------------
    obs[16] = float(n_nodes / 50.0)

    return obs
