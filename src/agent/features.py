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

# All 18 whiskers share the same half-angle (π/18), so cos_effective depends
# only on the body node, not the whisker index.
_HALF_ANGLE: float = math.pi / 18
_COS_HALF_ANGLE: float = math.cos(_HALF_ANGLE)
_SIN_HALF_ANGLE: float = math.sin(_HALF_ANGLE)
_TWO_NODE_ANGLE: float = 2.0 * NODE_ANGLE
# Whisker offsets as a float64 ndarray for broadcasting: shape (18,).
_OFFSETS_ARR: np.ndarray = np.array(_WHISKER_OFFSETS, dtype=np.float64)


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
    #
    # For each body node, compute the great-circle arc to its center and
    # the effective cone threshold that accounts for apparent angular width
    # at close range.  Then broadcast 18 ray directions against all M body
    # nodes to test hit/miss and find the nearest hit per whisker.
    #
    # cos_effective uses the identity cos(h + asin(t)) = cos(h)√(1-t²) − sin(h)·t
    # (where t = SIN_NODE_ANGLE/sin_arc) to avoid calling asin and cos per node.
    # sin(arc_to_center) = √(1-bz²) avoids a sin() call.
    # arc_to_center ≤ π/2 ⟺ bz ≤ 0, avoiding an acos() call for the hemisphere test.
    # ------------------------------------------------------------------
    n_nodes = len(snake)
    n_body = n_nodes - 2

    if n_body > 0:
        body = snake[2:]  # (M, 3)
        bx = body[:, 0]  # (M,)
        by = body[:, 1]  # (M,)
        bz = body[:, 2]  # (M,)

        n2d = np.sqrt(bx * bx + by * by)  # (M,) — projected XY length
        valid_n2d = n2d >= 1e-9  # (M,) — False for nodes at the south pole

        arc = np.arccos(np.clip(-bz, -1.0, 1.0))  # (M,) — great-circle distance to each node
        sin_arc = np.sqrt(np.maximum(0.0, 1.0 - bz * bz))  # (M,) — sin(arc), via identity

        near = bz <= 0.0  # (M,) — node in near hemisphere (arc ≤ π/2)
        too_small = sin_arc < _SIN_NODE_ANGLE  # (M,) — node so close it subtends the full sphere

        # Apparent-width threshold: cos(half_angle + apparent_half_width).
        # Guard denominator for nodes where sin_arc < SIN_NODE_ANGLE (handled by too_small).
        sin_arc_safe = np.where(sin_arc > _SIN_NODE_ANGLE, sin_arc, 1.0)
        t = _SIN_NODE_ANGLE / sin_arc_safe
        cos_eff_near = _COS_HALF_ANGLE * np.sqrt(np.maximum(0.0, 1.0 - t * t)) - _SIN_HALF_ANGLE * t
        cos_eff_near = np.where(too_small, -1.0, cos_eff_near)  # -1 means hits every cone
        cos_effective = np.where(near, cos_eff_near, _COS_HALF_ANGLE)  # (M,)

        # 18 ray directions from the current heading.
        alphas = direction + _OFFSETS_ARR  # (18,)
        ray_x = -np.cos(alphas)  # (18,)
        ray_y = -np.sin(alphas)  # (18,)

        # cos_lat[wi, j]: cosine between whisker wi's ray and the XY direction of node j.
        n2d_safe = np.where(valid_n2d, n2d, 1.0)
        cos_lat = (bx * ray_x[:, np.newaxis] + by * ray_y[:, np.newaxis]) / n2d_safe  # (18, M)

        # Surface-to-surface arc clearance for each (whisker, node) pair.
        arc_vals = np.maximum(0.0, arc * cos_lat - _TWO_NODE_ANGLE)  # (18, M)

        # A node hits a whisker if its projected direction is inside the cone and it is valid.
        hits = (cos_lat >= cos_effective) & valid_n2d  # (18, M)

        # Nearest hit per whisker; _MAX_DIST is the sentinel for no-hit.
        min_arcs = np.where(hits, arc_vals, _MAX_DIST).min(axis=1)  # (18,)
    else:
        min_arcs = np.full(18, _MAX_DIST, dtype=np.float64)

    obs[3:21] = (1.0 - min_arcs / _MAX_DIST).astype(np.float32)

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
