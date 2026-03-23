"""
Feature extraction for SphericalSnakeEnv.

    compute_obs(snake, pellet, direction) -> np.ndarray[float32, (25,)]

Observation layout:
  Index | Feature
  ------|---------------------------------------------------------
  0     | pellet_bearing_sin  — sin of signed lateral angle head→pellet
  1     | pellet_bearing_cos  — cos of same (avoids ±π discontinuity)
  2     | pellet_dist         — great-circle distance head→pellet / π
  3-20  | whiskers[18]        — 18 rays; 0=safe, 1=imminent collision
            Uniform 20° cones (10° half-angle), 20° spacing, full 360° coverage.
            0° (front), ±20°, ±40°, ±60°, ±80°, ±100°, ±120°, ±140°, ±160°, 180° (back).
  21    | head_z              — z coordinate of head
  22    | sin(direction)
  23    | cos(direction)
  24    | snake_len_norm       — len(snake) / 50
"""

import math
import numpy as np

NODE_ANGLE: float = math.pi / 60  # angular radius of one snake node
_SIN_NODE_ANGLE: float = math.sin(NODE_ANGLE)  # used for apparent-width cone expansion

# Whisker ray offsets from heading angle, indices 3-20.
# 18 whiskers at 20° spacing, uniform 20° cones (10° half-angle each).
# One exactly at 0° (front) and one at 180° (back); symmetric ± pairs in between.
# Coverage: 18 × 20° = 360° with zero gaps and zero overlaps.
_WHISKER_OFFSETS: tuple = (
    0,  #   0° front  (index 3)
    math.pi / 9,  # +20°        (index 4)
    -math.pi / 9,  # -20°        (index 5)
    2 * math.pi / 9,  # +40°        (index 6)
    -2 * math.pi / 9,  # -40°        (index 7)
    math.pi / 3,  # +60°        (index 8)
    -math.pi / 3,  # -60°        (index 9)
    4 * math.pi / 9,  # +80°        (index 10)
    -4 * math.pi / 9,  # -80°        (index 11)
    5 * math.pi / 9,  # +100°       (index 12)
    -5 * math.pi / 9,  # -100°       (index 13)
    2 * math.pi / 3,  # +120°       (index 14)
    -2 * math.pi / 3,  # -120°       (index 15)
    7 * math.pi / 9,  # +140°       (index 16)
    -7 * math.pi / 9,  # -140°       (index 17)
    8 * math.pi / 9,  # +160°       (index 18)
    -8 * math.pi / 9,  # -160°       (index 19)
    math.pi,  # 180° back   (index 20)
)

# Per-whisker half-angles: all π/18 (10°) — uniform 20° total cone width.
_WHISKER_HALF_ANGLES: tuple = tuple(math.pi / 18 for _ in _WHISKER_OFFSETS)
_WHISKER_COS_HALF: tuple = tuple(math.cos(h) for h in _WHISKER_HALF_ANGLES)

# Maximum great-circle distance used to normalise whisker and pellet readings.
_MAX_DIST: float = math.pi


def compute_obs(
    snake: np.ndarray,
    pellet: np.ndarray,
    direction: float,
) -> np.ndarray:
    """
    Return the 25-element float32 observation vector.

    Parameters
    ----------
    snake      : (N, 3) float64 — node positions; snake[0] is the head,
                 invariantly at (0, 0, -1) under the world-rotation scheme.
    pellet     : (3,) float64 — unit-sphere food position.
    direction  : float — current heading angle (radians).
    """
    obs = np.empty(25, dtype=np.float32)

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
    # Indices 3-20 : whiskers
    # ------------------------------------------------------------------
    n_nodes = len(snake)

    for wi, offset in enumerate(_WHISKER_OFFSETS):
        alpha = direction + offset
        ray_x = -math.cos(alpha)
        ray_y = -math.sin(alpha)
        half_angle = _WHISKER_HALF_ANGLES[wi]
        min_arc = _MAX_DIST

        for j in range(2, n_nodes):
            # Tangent-plane projection at (0,0,-1) is just the xy components.
            bx = float(snake[j, 0])
            by = float(snake[j, 1])
            bz = float(snake[j, 2])

            n2d = math.sqrt(bx * bx + by * by)
            if n2d < 1e-9:
                continue  # node is at the south pole (on top of head)

            # Cosine of angle between node's projected direction and ray.
            cos_lat = (bx * ray_x + by * ray_y) / n2d

            # Dynamic apparent-width cone expansion: at close range a segment's
            # angular half-size asin(sin(NODE_ANGLE)/sin(d)) >> NODE_ANGLE, so
            # a fixed threshold misses segments whose center is just outside the
            # nominal cone while their body fully blocks the path.
            arc_to_center = math.acos(max(-1.0, min(1.0, -bz)))

            # Apparent-width expansion: only within the near hemisphere (arc ≤ π/2).
            # Beyond that, a segment cannot physically overlap a cone edge any more
            # than at nominal size, and the formula diverges near the antipodal point.
            if arc_to_center <= math.pi / 2:
                sin_arc = math.sin(arc_to_center)
                if sin_arc < _SIN_NODE_ANGLE:
                    cos_effective = -1.0  # segment on the head: hits all cones
                else:
                    apparent_hw = math.asin(_SIN_NODE_ANGLE / sin_arc)
                    cos_effective = math.cos(half_angle + apparent_hw)
            else:
                cos_effective = math.cos(half_angle)  # original nominal threshold

            if cos_lat < cos_effective:
                continue  # segment body does not reach into this whisker's cone

            # Projected arc: arc_to_center * cos_lat gives forward-direction
            # clearance; subtract 2*NODE_ANGLE for surface-to-surface gap.
            arc = max(0.0, arc_to_center * cos_lat - 2.0 * NODE_ANGLE)
            if arc < min_arc:
                min_arc = arc

        obs[3 + wi] = float(1.0 - min_arc / _MAX_DIST)

    # ------------------------------------------------------------------
    # Index 21 : head z coordinate
    # ------------------------------------------------------------------
    obs[21] = float(head[2])

    # ------------------------------------------------------------------
    # Indices 22-23 : sin/cos of direction
    # ------------------------------------------------------------------
    obs[22] = float(sin_d)
    obs[23] = float(cos_d)

    # ------------------------------------------------------------------
    # Index 24 : snake length normalised
    # ------------------------------------------------------------------
    obs[24] = float(n_nodes / 50.0)

    return obs
