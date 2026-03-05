import math
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces


NODE_ANGLE: float = math.pi / 60
NODE_QUEUE_SIZE: int = 9
STARTING_DIRECTION: float = math.pi / 4
COLLISION_DISTANCE: float = 2.0 * math.sin(NODE_ANGLE)
SNAKE_VELOCITY: float = NODE_ANGLE * 2.0 / (NODE_QUEUE_SIZE + 1)
TURN_RATE: float = 0.08
INITIAL_SNAKE_LENGTH: int = 8


def rotate_z(a: float, pts: np.ndarray) -> None:
    """rotateZ(a) in-place. Accepts (3,) or (N, 3)."""
    p = pts[np.newaxis] if pts.ndim == 1 else pts
    cos_a, sin_a = math.cos(a), math.sin(a)
    x = p[:, 0].copy()  # must copy: x is overwritten before being used in row 2
    p[:, 0] = cos_a * x - sin_a * p[:, 1]
    p[:, 1] = sin_a * x + cos_a * p[:, 1]


def rotate_y(a: float, pts: np.ndarray) -> None:
    """rotateY(a) in-place — JS non-standard convention. Accepts (3,) or (N, 3)."""
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
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32
        )
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
        obs = self._compute_obs()
        return obs, {}

    def step(self, action: int):
        assert not self._terminated, "Episode has ended; call reset() first."

        prev_dist = self._angular_dist_to_pellet()

        # --- 1. Check collisions on current positions (JS tick order) ------
        eaten, self_collision = self._check_collisions()

        if self_collision:
            self._terminated = True
            obs = self._compute_obs()
            return obs, -1.0, True, False, {"score": self.score}

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

        obs = self._compute_obs()
        return obs, float(reward), False, False, {"score": self.score}

    def _regenerate_pellet(self) -> np.ndarray:
        """
        Mirrors JS regeneratePellet() / pointFromSpherical().
        Uniform random (theta, phi) — intentionally non-uniform on sphere surface.
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
        Mirrors JS addSnakeNode().
        New node starts with posQueue filled with None (NODE_QUEUE_SIZE entries).
        Position is determined by the tail of the last node's posQueue, or by
        rotating backward using STARTING_DIRECTION if that entry is still None.
        First node ever is placed at the north pole (0, 0, 1).
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

        self.snake = (
            pos[np.newaxis, :] if len(self.snake) == 0
            else np.vstack([self.snake, pos[np.newaxis, :]])
        )
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
        Mirrors JS checkCollisions().
        Returns (pellet_eaten, self_collision).

        Self-collision: head vs snake[i] for i >= 2 (index 1 is always adjacent).
        Pellet collision triggers regeneratePellet() + addSnakeNode() + score++.
        Self-collision takes priority — if both happen, returns (False, True).
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
        """Great-circle angular distance head → pellet, in [0, π]."""
        dot = float(np.clip(np.dot(self.snake[0], self.pellet), -1.0, 1.0))
        return math.acos(dot)

    def _compute_obs(self) -> np.ndarray:
        """Delegate to features.compute_obs (implemented in Step 2)."""
        from agent.features import compute_obs  # lazy import; stub until Step 2
        return compute_obs(self.snake, self.pos_queues, self.pellet, self.direction)
