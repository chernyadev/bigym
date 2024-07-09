"""Set of fridge tasks."""
from abc import ABC

import numpy as np

from bigym.bigym_env import BiGymEnv
from bigym.const import PRESETS_PATH
from bigym.envs.props.fridge import Fridge


class _FridgeEnv(BiGymEnv, ABC):
    """Base fridge environment."""

    RESET_ROBOT_POS = np.array([0, -0.8, 0])

    _PRESET_PATH = PRESETS_PATH / "fridge.yaml"
    _TOLERANCE = 0.05

    def _initialize_env(self):
        self.dishwasher = self._preset.get_props(Fridge)[0]


class OpenFridge(_FridgeEnv):
    """Open fridge door."""
    pass
