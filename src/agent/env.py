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
    observation_space: Box(shape=(15,), dtype=float32)  — see features.py
    action_space:      Discrete(3)  — 0=STRAIGHT, 1=LEFT, 2=RIGHT
    """

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

        # State (initialised properly in reset())
        # snake mirrors JS: shape (N, 3) of node positions; pos_queues[i] mirrors snake[i].posQueue
        self.snake: np.ndarray = np.zeros((INITIAL_SNAKE_LENGTH, 3), dtype=np.float64)
        self.pos_queues: list[list] = []  # pos_queues[i] = NODE_QUEUE_SIZE history entries for snake[i]
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
        self.pos_queues = []
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

        The new node's posQueue is initialised to [None]*NODE_QUEUE_SIZE.
        Its starting position is:
          • (0, 0, -1) if this is the very first node (south pole).
          • self.pos_queues[-1][-1] if that queue entry is not None
            (the preceding tail's oldest position history entry).
          • Otherwise the fallback: rotate the current tail position
            backward one step using STARTING_DIRECTION (matches JS behaviour
            during the first NODE_QUEUE_SIZE ticks of a node's life).

        Modifies self.snake (extended by one row) and self.pos_queues in place.
        """
        queue: list = [None] * NODE_QUEUE_SIZE

        if len(self.snake) == 0:
            # First node: south pole — matches JS addSnakeNode() default {x:0, y:0, z:-1}
            pos = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        else:
            last_pos = self.snake[-1].copy()
            last_queue_tail = self.pos_queues[-1][NODE_QUEUE_SIZE - 1]

            if last_queue_tail is None:
                # Fallback — mirrors JS: rotate "behind" last node
                pos = last_pos.copy()
                rotate_z(-STARTING_DIRECTION, pos)
                rotate_y(-NODE_ANGLE * 2.0, pos)
                rotate_z(STARTING_DIRECTION, pos)
            else:
                pos = last_queue_tail.copy()

        self.snake = pos[np.newaxis, :] if len(self.snake) == 0 else np.vstack([self.snake, pos[np.newaxis, :]])
        self.pos_queues.append(queue)

    def _apply_snake_rotation(self) -> None:
        """
        Mirrors JS applySnakeRotation() exactly.

        For each node i (0 → N-1):
          1. Save old_position = copy of current snake[i]
          2. Move node:
             - i == 0 (head):           rotateZ(-dir) → rotateY(+vel) → rotateZ(+dir)
             - i > 0, next_pos is None: rotateZ(-STARTING_DIR) → rotateY(+vel) → rotateZ(+STARTING_DIR)
             - i > 0, next_pos is set:  teleport to next_pos
          3. posQueue.unshift(old_position)   [prepend]
          4. next_pos = posQueue.pop()        [pop from back — oldest entry]
        """
        next_position = None

        for i in range(len(self.snake)):
            old_position = self.snake[i].copy()

            if i == 0:
                pt = self.snake[i].copy()
                rotate_z(-self.direction, pt)
                rotate_y(SNAKE_VELOCITY, pt)
                rotate_z(self.direction, pt)
                self.snake[i] = pt
            elif next_position is None:
                # Body node whose predecessor has no history yet — use fallback
                pt = self.snake[i].copy()
                rotate_z(-STARTING_DIRECTION, pt)
                rotate_y(SNAKE_VELOCITY, pt)
                rotate_z(STARTING_DIRECTION, pt)
                self.snake[i] = pt
            else:
                self.snake[i] = next_position

            # posQueue.unshift(old) → insert at index 0
            self.pos_queues[i].insert(0, old_position)
            # posQueue.pop() → remove last element; feeds next node
            next_position = self.pos_queues[i].pop()

    def _world_rotation(self) -> None:
        """
        World rotation that mirrors JS: rotateZ(-d) → rotateY(-v) → rotateZ(+d)
        applied to ALL points: pellet, every snake node, every non-None posQueue entry.
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

        for queue in self.pos_queues:
            for j in range(len(queue)):
                if queue[j] is not None:
                    rotate_z(-d, queue[j])
                    rotate_y(-v, queue[j])
                    rotate_z(d, queue[j])

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

        # Self-collision check first (mirrors JS early-return on showEnd())
        for i in range(2, len(self.snake)):
            diff = head - self.snake[i]
            if math.sqrt(float(np.dot(diff, diff))) < COLLISION_DISTANCE:
                return False, True

        # Pellet collision
        diff = head - self.pellet
        eaten = math.sqrt(float(np.dot(diff, diff))) < COLLISION_DISTANCE
        if eaten:
            self.pellet = self._regenerate_pellet()
            self._add_snake_node()
            self.score += 1

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
