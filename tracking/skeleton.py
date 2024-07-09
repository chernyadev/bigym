"""Skeleton for H1 Control."""
from abc import ABC, abstractmethod


class Skeleton(ABC):
    """Skeleton required for H1 Control."""

    def __init__(self):
        """Initialize H1 Control State."""
        self._joint_ref_tfs = {}
        self._skeleton = {}
        self._ref_skeleton = {}

    @abstractmethod
    def update(self):
        """Update skeleton positions."""
        pass

    @property
    def skeleton(self):
        """Get skeleton."""
        return self._skeleton

    @property
    def reference_skeleton(self):
        """Get reference skeleton."""
        return self._ref_skeleton
