"""
Feature extraction for SphericalSnakeEnv.

    compute_obs(snake, pos_queues, pellet, direction) -> np.ndarray[float32, (16,)]

Observation layout:
  Index | Feature
  ------|---------------------------------------------------------
  0     | pellet_bearing_sin  — sin of signed lateral angle head→pellet
  1     | pellet_bearing_cos  — cos of same (avoids ±π discontinuity)
  2     | pellet_dist         — great-circle distance head→pellet / π
  3-10  | whiskers[8]         — 8 rays; 0=safe, 1=imminent collision
  11    | head_z              — z coordinate of head
  12    | sin(direction)
  13    | cos(direction)
  14    | snake_len_norm       — len(snake) / 50
"""

import math
import numpy as np


# Whisker ray offsets from heading angle, indices 3-10
_WHISKER_OFFSETS: tuple = (
    math.pi / 8,  # +22.5°  (index 3)
    -math.pi / 8,  # -22.5°  (index 4)
    3 * math.pi / 8,  # +67.5°  (index 5)
    -3 * math.pi / 8,  # -67.5°  (index 6)
    5 * math.pi / 8,  # +112.5° (index 7)
    -5 * math.pi / 8,  # -112.5° (index 8)
    7 * math.pi / 8,  # +157.5° (index 9)
    -7 * math.pi / 8,  # -157.5° (index 10)
)

# Cosine of the whisker half-angle (22.5°).  A body node is inside the cone
# when its alignment dot-product with the ray direction exceeds this value.
_COS_HALF: float = math.cos(math.pi / 8)

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
    obs = np.empty(15, dtype=np.float32)

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
            if cos_lat < _COS_HALF:
                continue  # node is outside this whisker's 45° cone

            # Great-circle arc distance: dot(b, head) = -bz for head=(0,0,-1).
            arc = math.acos(max(-1.0, min(1.0, -bz)))
            if arc < min_arc:
                min_arc = arc

        obs[3 + wi] = float(1.0 - min_arc / _MAX_DIST)

    # ------------------------------------------------------------------
    # Index 11 : head z coordinate
    # ------------------------------------------------------------------
    obs[11] = float(head[2])

    # ------------------------------------------------------------------
    # Indices 12-13 : sin/cos of direction
    # ------------------------------------------------------------------
    obs[12] = float(sin_d)
    obs[13] = float(cos_d)

    # ------------------------------------------------------------------
    # Index 14 : snake length normalised
    # ------------------------------------------------------------------
    obs[14] = float(n_nodes / 50.0)

    return obs
