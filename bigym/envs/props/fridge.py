"""Fridge."""
from pathlib import Path

from bigym.const import ASSETS_PATH
from bigym.envs.props.prop import Prop


class Fridge(Prop):
    """Fridge."""

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/dishwasher/dishwasher.xml"
