import math
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from agent.features import compute_obs, NODE_ANGLE

NODE_QUEUE_SIZE: int = 9
STARTING_DIRECTION: float = math.pi / 4
COLLISION_DISTANCE: float = 2.0 * math.sin(NODE_ANGLE)
SNAKE_VELOCITY: float = NODE_ANGLE * 2.0 / (NODE_QUEUE_SIZE + 1)
TURN_RATE: float = 0.08
INITIAL_SNAKE_LENGTH: int = 8

# Curriculum learning: arc-distance bounds for near-pellet placement.
# The minimum is slightly above the collision threshold so the pellet is
# not eaten on the very next tick; the maximum keeps it reachable within
# a few dozen steps from the head (which is always at the south pole).
_NEAR_PELLET_MIN_ARC: float = COLLISION_DISTANCE + 0.01  # ~0.115 rad
_NEAR_PELLET_MAX_ARC: float = 0.35  # ~20° arc radius
_COLLISION_DIST_SQ: float = COLLISION_DISTANCE**2


def rotate_z(a: float, pts: np.ndarray) -> None:
    """
    Apply rotateZ(a) in-place, mirroring the JS rotateZ convention.

    Parameters
    ----------
    a   : rotation angle in radians.
    pts : array of shape (3,) or (N, 3) — modified in place.
    """
    p = pts[np.newaxis] if pts.ndim == 1 else pts
    cos_a, sin_a = math.cos(a), math.sin(a)
    x = p[:, 0].copy()  # must copy: x is overwritten before being used in row 2
    p[:, 0] = cos_a * x - sin_a * p[:, 1]
    p[:, 1] = sin_a * x + cos_a * p[:, 1]


def rotate_y(a: float, pts: np.ndarray) -> None:
    """
    Apply rotateY(a) in-place, mirroring the JS rotateY convention.

    Note: the JS game uses a non-standard rotateY where the sign of the
    x↔z cross-terms is swapped relative to the standard right-hand rule.
    This implementation matches that convention exactly.

    Parameters
    ----------
    a   : rotation angle in radians.
    pts : array of shape (3,) or (N, 3) — modified in place.
    """
    p = pts[np.newaxis] if pts.ndim == 1 else pts
    cos_a, sin_a = math.cos(a), math.sin(a)
    x = p[:, 0].copy()  # must copy: x is overwritten before being used in row 2
    p[:, 0] = cos_a * x + sin_a * p[:, 2]
    p[:, 2] = -sin_a * x + cos_a * p[:, 2]


class SphericalSnakeEnv(gym.Env):
    """
    observation_space: Box(shape=(25,), dtype=float32)  — see features.py
    action_space:      Discrete(3)  — 0=STRAIGHT, 1=LEFT, 2=RIGHT
    """

    metadata = {"render_modes": []}

    def __init__(self, curriculum_length: int = 0) -> None:
        """
        Parameters
        ----------
        curriculum_length : int
            Number of pellets per episode that are placed close to the head
            (within ``_NEAR_PELLET_MAX_ARC`` arc distance) to guarantee the
            snake reaches a high body-count state quickly.  Once
            ``self.score >= curriculum_length`` pellets spawn uniformly at
            random (the original behaviour).  Set to 0 (default) to disable
            curriculum entirely and keep full backward compatibility.
        """
        super().__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(25,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.curriculum_length: int = curriculum_length

        # State (initialised properly in reset())
        # snake: (N, 3) node positions; snake[0] is the head.
        # pos_queues: (N, NODE_QUEUE_SIZE, 3) contiguous history buffer.
        #   pos_queues[i, 0] = most-recent saved position of node i.
        #   pos_queues[i, -1] = oldest saved position → fed to node i+1 as teleport target.
        self.snake: np.ndarray = np.zeros((INITIAL_SNAKE_LENGTH, 3), dtype=np.float64)
        self.pos_queues: np.ndarray = np.empty((0, NODE_QUEUE_SIZE, 3), dtype=np.float64)
        self.pellet: np.ndarray = np.zeros(3, dtype=np.float64)
        self.direction: float = STARTING_DIRECTION
        self.score: int = 0
        self._terminated: bool = False

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        """
        Reset the environment to its initial state.

        Spawns a fresh 8-node snake at the south pole with direction=π/4,
        then places a random pellet via _regenerate_pellet().

        Parameters
        ----------
        seed    : optional RNG seed passed to gymnasium's np_random.
        options : unused; accepted for API compatibility.

        Returns
        -------
        obs  : np.ndarray, shape (16,), dtype float32 — initial observation.
        info : dict — empty dict (no extra info on reset).
        """
        super().reset(seed=seed)

        self.direction = STARTING_DIRECTION
        self.score = 0
        self._terminated = False

        # Build initial snake (8 nodes) — mirrors JS init() calling addSnakeNode() × 8
        self.snake = np.empty((0, 3), dtype=np.float64)
        self.pos_queues = np.empty((0, NODE_QUEUE_SIZE, 3), dtype=np.float64)
        for _ in range(INITIAL_SNAKE_LENGTH):
            self._add_snake_node()

        self.pellet = self._regenerate_pellet()
        obs = compute_obs(self.snake, self.pellet, self.direction)
        return obs, {}

    def step(self, action: int):
        """
        Advance the simulation by one tick.

        Tick order mirrors the JS game:
          1. Check collisions on the *current* positions.
          2. Update direction based on action.
          3. Move snake nodes (applySnakeRotation).
          4. Apply world rotation to all points.
          5. Compute reward and next observation.

        Parameters
        ----------
        action : int — 0=STRAIGHT, 1=LEFT (direction-=0.08), 2=RIGHT (direction+=0.08).

        Returns
        -------
        obs        : np.ndarray, shape (16,), dtype float32.
        reward     : float — +1.0 pellet eaten, -1.0 self-collision,
                     +0.05*(prev_dist-new_dist) progress shaping, -0.001 time penalty.
        terminated : bool — True if the snake hit itself.
        truncated  : bool — always False (no time limit).
        info       : dict with key 'score' (int, pellets eaten this episode).
        """
        assert not self._terminated, "Episode has ended; call reset() first."

        prev_dist = self._angular_dist_to_pellet()

        # --- 1. Check collisions on current positions (JS tick order) ------
        eaten, self_collision = self._check_collisions()

        if self_collision:
            self._terminated = True
            obs = compute_obs(self.snake, self.pellet, self.direction)
            return obs, -10.0, True, False, {"score": self.score}

        # --- 2. Apply action → update direction ----------------------------
        if action == 1:
            self.direction -= TURN_RATE
        elif action == 2:
            self.direction += TURN_RATE
        # action == 0: straight, no change

        # --- 3. Move snake nodes -------------------------------------------
        self._apply_snake_rotation()

        # --- 4. World rotation on ALL points --------------------------------
        self._world_rotation()

        # --- 5. Compute reward ---------------------------------------------
        new_dist = self._angular_dist_to_pellet()
        reward = (1.0 if eaten else 0.0) + 0.05 * (prev_dist - new_dist) - 0.001

        obs = compute_obs(self.snake, self.pellet, self.direction)
        return obs, float(reward), False, False, {"score": self.score}

    def _place_nearby_pellet(self) -> np.ndarray:
        """
        Place a pellet close to the head for curriculum learning.

        Samples a point on the unit sphere that is between
        ``_NEAR_PELLET_MIN_ARC`` and ``_NEAR_PELLET_MAX_ARC`` great-circle
        arc distance from the head.

        Returns
        -------
        np.ndarray, shape (3,), dtype float64 — unit-sphere Cartesian point.
        """
        arc = self.np_random.uniform(_NEAR_PELLET_MIN_ARC, _NEAR_PELLET_MAX_ARC)
        bearing = self.np_random.uniform(0.0, 2.0 * math.pi)
        sin_arc = math.sin(arc)
        return np.array(
            [sin_arc * math.cos(bearing), sin_arc * math.sin(bearing), -math.cos(arc)],
            dtype=np.float64,
        )

    def _regenerate_pellet(self) -> np.ndarray:
        """
        Mirrors JS regeneratePellet() / pointFromSpherical().

        Samples (theta, phi) uniformly — intentionally non-uniform on the sphere
        surface (no sin(φ) density correction), matching the JS distribution.

        Returns
        -------
        np.ndarray, shape (3,), dtype float64 — unit-sphere Cartesian point.
        """
        theta = self.np_random.uniform(0.0, 2.0 * math.pi)
        phi = self.np_random.uniform(0.0, math.pi)
        sin_phi = math.sin(phi)
        return np.array(
            [math.cos(theta) * sin_phi, math.sin(theta) * sin_phi, math.cos(phi)],
            dtype=np.float64,
        )

    def _add_snake_node(self) -> None:
        """
        Append a new tail node to self.snake, mirroring JS addSnakeNode().

        Position: (0, 0, -1) for the first node; otherwise taken from
        pos_queues[-1, -1] (the oldest queue slot of the current tail).

        The new node's queue is pre-filled with a fallback position: the new
        node's position rotated one step "behind" using STARTING_DIRECTION.
        This exactly replicates the JS null-entry path: _apply_snake_rotation
        will place the node-after-this-one at pos_queues[-1, -1] = fallback,
        which is identical to what the original None-branch computes.
        """
        if len(self.snake) == 0:
            pos = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        else:
            # pos_queues[-1, -1] is always a valid float64: pre-filled or real history.
            pos = self.pos_queues[-1, NODE_QUEUE_SIZE - 1].copy()

        self.snake = pos[np.newaxis, :] if len(self.snake) == 0 else np.vstack([self.snake, pos[np.newaxis, :]])

        # Pre-fill this node's queue with the one-step-behind fallback position.
        # Mirrors the JS null-entry fallback used during the first NODE_QUEUE_SIZE ticks.
        fallback = pos.copy()
        rotate_z(-STARTING_DIRECTION, fallback)
        rotate_y(-NODE_ANGLE * 2.0, fallback)
        rotate_z(STARTING_DIRECTION, fallback)
        # Broadcast fallback across all NODE_QUEUE_SIZE slots → shape (1, 9, 3)
        new_row = np.broadcast_to(fallback, (NODE_QUEUE_SIZE, 3)).copy()[np.newaxis]
        self.pos_queues = np.concatenate([self.pos_queues, new_row], axis=0)

    def _apply_snake_rotation(self) -> None:
        """
        Mirrors JS applySnakeRotation().

        Head node rotates; each body node teleports to the tail of its
        predecessor's position queue. All queues right-shift and receive
        the node's pre-rotation position at the front.
        """
        # Save originals needed after pos_queues is right-shifted.
        old_head = self.snake[0].copy()  # pushed to pos_queues[0, 0]
        old_body = self.snake[1:].copy()  # pushed to pos_queues[1:, 0]; shape (N-1, 3)

        # Rotate head in place (three in-place rotations on the (3,) slice).
        rotate_z(-self.direction, self.snake[0])
        rotate_y(SNAKE_VELOCITY, self.snake[0])
        rotate_z(self.direction, self.snake[0])

        # Teleport each body node to pos_queues[i-1, -1] (read before right-shift).
        self.snake[1:] = self.pos_queues[:-1, -1]

        # Right-shift all queues simultaneously (oldest entry at slot 8 drops off).
        self.pos_queues[:, 1:] = self.pos_queues[:, :-1].copy()

        # Push pre-rotation positions to the queue front.
        self.pos_queues[0, 0] = old_head
        self.pos_queues[1:, 0] = old_body

    def _world_rotation(self) -> None:
        """
        World rotation that mirrors JS: rotateZ(-d) → rotateY(-v) → rotateZ(+d)
        applied to ALL points: pellet, every snake node, every posQueue entry.
        Grid points (points[] in JS) are rendering-only and excluded here.
        """
        d = self.direction
        v = SNAKE_VELOCITY

        rotate_z(-d, self.pellet)
        rotate_y(-v, self.pellet)
        rotate_z(d, self.pellet)

        rotate_z(-d, self.snake)
        rotate_y(-v, self.snake)
        rotate_z(d, self.snake)

        flat = self.pos_queues.reshape(-1, 3)  # view: (N*9, 3)
        rotate_z(-d, flat)
        rotate_y(-v, flat)
        rotate_z(d, flat)

    def _check_collisions(self) -> tuple[bool, bool]:
        """
        Check pellet and self-collision for the current frame, mirroring JS checkCollisions().

        Self-collision is tested against snake[i] for i ≥ 2 (node 1 is always
        adjacent to the head and cannot collide). If a self-collision is detected
        the pellet check is skipped.

        A pellet collision triggers _regenerate_pellet(), _add_snake_node(), and
        increments self.score.

        Returns
        -------
        pellet_eaten   : bool — True if the head overlapped the pellet this tick.
        self_collision : bool — True if the head overlapped any body node i ≥ 2.
        """
        head = self.snake[0]

        # Self-collision: compare squared chord distances to avoid sqrt.
        # snake[1] is always adjacent and cannot collide; test snake[2:].
        if len(self.snake) > 2:
            diffs = self.snake[2:] - head  # (N-2, 3)
            sq_dists = np.einsum("ij,ij->i", diffs, diffs)  # (N-2,)
            if bool(np.any(sq_dists < _COLLISION_DIST_SQ)):
                return False, True

        # Pellet collision (squared distance, no sqrt)
        diff = head - self.pellet
        eaten = float(np.dot(diff, diff)) < _COLLISION_DIST_SQ
        if eaten:
            self.score += 1
            if self.curriculum_length > 0 and self.score < self.curriculum_length:
                self.pellet = self._place_nearby_pellet()
            else:
                self.pellet = self._regenerate_pellet()
            self._add_snake_node()

        return eaten, False

    def _angular_dist_to_pellet(self) -> float:
        """
        Compute the great-circle angular distance from the head to the pellet.

        Returns
        -------
        float — angle in radians, range [0, π].
        """
        dot = float(np.clip(np.dot(self.snake[0], self.pellet), -1.0, 1.0))
        return math.acos(dot)
