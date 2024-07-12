"""Set of reach target tasks."""
from abc import ABC
from dataclasses import dataclass, field

import numpy as np
from gymnasium import spaces
from mojo import Mojo
from mojo.elements import Body, Geom
from mojo.elements.consts import GeomType

from bigym.bigym_env import BiGymEnv
from bigym.const import HandSide


@dataclass
class TargetConfig:
    """Target Config."""

    reset_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    size: np.ndarray = field(default_factory=lambda: np.array([0.05, 0.05, 0.05]))
    color_default: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 1]))
    color_highlight: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 1]))


class Target:
    """Target sphere."""

    def __init__(self, mojo: Mojo, config: TargetConfig):
        """Init."""
        self._config = config
        self.body = Body.create(mojo)
        self.geom: Geom = Geom.create(
            mojo,
            parent=self.body,
            geom_type=GeomType.SPHERE,
            size=config.size,
            color=config.color_default,
            mass=0,
        )
        self.geom.set_collidable(False)

    def reset_position(self, offset: np.ndarray = np.zeros(3)):
        """Reset position of the target."""
        self.body.set_position(self._config.reset_position + offset)

    def distance(self, pos: np.ndarray) -> float:
        """Get distance to target."""
        return float(np.linalg.norm(self.body.get_position() - pos))

    def is_reached(
        self, pos: np.ndarray, tolerance: float, highlight: bool = False
    ) -> bool:
        """Check if target is reached."""
        is_reached = self.distance(pos) <= tolerance
        if highlight:
            self.set_highlight(is_reached)
        return is_reached

    def set_highlight(self, highlight: bool):
        """Toggle target highlight."""
        self.geom.set_color(
            self._config.color_highlight if highlight else self._config.color_default
        )


class _ReachTargetEnv(BiGymEnv, ABC):
    """Base reach target environment."""

    TARGET_CONFIGS = [
        TargetConfig(
            reset_position=np.array([0.5, 0, 1]),
            color_default=np.array([1, 0, 0, 1]),
            color_highlight=np.array([1, 0, 0, 1]),
        )
    ]

    POSITION_BOUNDS = np.array([0.1, 0.1, 0.1])
    TOLERANCE = 0.1

    def _initialize_env(self):
        self.targets: list[Target] = []
        for config in self.TARGET_CONFIGS:
            self.targets.append(Target(self._mojo, config))

    def _on_reset(self):
        for target in self.targets:
            offset = np.random.uniform(-self.POSITION_BOUNDS, self.POSITION_BOUNDS)
            target.reset_position(offset)


class ReachTarget(_ReachTargetEnv):
    """Reach the target with either left or right wrist."""

    def _get_task_privileged_obs_space(self):
        return {
            "target_position": spaces.Box(
                low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
            )
        }

    def _get_task_privileged_obs(self):
        return {
            "target_position": np.array(
                self.targets[0].body.get_position(), np.float32
            ).copy()
        }

    def _success(self) -> bool:
        for side in self.robot.grippers:
            if self.targets[0].is_reached(
                self._robot.get_hand_pos(side),
                self.TOLERANCE,
            ):
                return True
        return False


class ReachTargetSingle(ReachTarget):
    """Reach the target with specific wrist."""

    SIDE = HandSide.LEFT

    def _initialize_env(self):
        super()._initialize_env()

    def _success(self) -> bool:
        return self.targets[0].is_reached(
            self._robot.get_hand_pos(self.SIDE), self.TOLERANCE
        )


class ReachTargetDual(_ReachTargetEnv):
    """Reach 2 targets, one with each arm."""

    TARGET_CONFIGS = [
        TargetConfig(
            reset_position=np.array([0.5, 0.2, 1]),
            color_default=np.array([0.3, 0, 0, 1]),
            color_highlight=np.array([1, 0, 0, 1]),
        ),
        TargetConfig(
            reset_position=np.array([0.5, -0.2, 1]),
            color_default=np.array([0, 0.3, 0, 1]),
            color_highlight=np.array([0, 1, 0, 1]),
        ),
    ]

    def _get_task_privileged_obs_space(self):
        return {
            "target_position_left": spaces.Box(
                low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
            ),
            "target_position_right": spaces.Box(
                low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
            ),
        }

    def _get_task_privileged_obs(self):
        return {
            "target_position_left": np.array(
                self.targets[0].body.get_position(), np.float32
            ).copy(),
            "target_position_right": np.array(
                self.targets[1].body.get_position(), np.float32
            ).copy(),
        }

    def _success(self) -> bool:
        if not self.targets[0].is_reached(
            self._robot.get_hand_pos(HandSide.LEFT), self.TOLERANCE
        ):
            return False
        if not self.targets[1].is_reached(
            self._robot.get_hand_pos(HandSide.RIGHT), self.TOLERANCE
        ):
            return False
        return True

    def _on_step(self):
        self.targets[0].is_reached(
            self._robot.get_hand_pos(HandSide.LEFT), self.TOLERANCE, True
        )
        self.targets[1].is_reached(
            self._robot.get_hand_pos(HandSide.RIGHT), self.TOLERANCE, True
        )
