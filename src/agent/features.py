"""
Feature extraction for SphericalSnakeEnv.

Stub — returns a zero observation vector until Step 2 is implemented.

Signature (do not change):
    compute_obs(snake, pos_queues, pellet, direction) -> np.ndarray[float32, (15,)]

Observation layout (implemented in Step 2):
  Index | Feature
  ------|---------------------------------------------------------
  0     | pellet_bearing_sin
  1     | pellet_bearing_cos
  2     | pellet_dist  (great-circle / π)
  3-10  | whiskers[8]  (8 rays; 0=safe, 1=imminent collision)
  11    | head_z
  12    | sin(direction)
  13    | cos(direction)
  14    | snake_len_norm  (len / 50)
"""

import numpy as np


def compute_obs(
    snake: np.ndarray,
    pos_queues: list,
    pellet: np.ndarray,
    direction: float,
) -> np.ndarray:
    """Stub: returns zero vector. Replace with full implementation in Step 2."""
    return np.zeros(15, dtype=np.float32)
