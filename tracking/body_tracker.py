"""Abstract class for body tracking systems."""
from abc import ABC, abstractmethod
import logging
import numpy as np


class BodyTracker(ABC):
    """Abstract class for body tracking systems."""

    def __init__(self, timed=False):
        """Initialize body tracker."""
        self._skeleton = None
        self._timed = timed
        self._capture_times = []
        self._tracking_times = []

    @abstractmethod
    def update(self):
        """Update body tracker."""
        pass

    @property
    def skeleton(self):
        """Get body."""
        return self._skeleton.skeleton

    @property
    def reference_skeleton(self):
        """Get reference body."""
        return self._skeleton.reference_skeleton

    def get_new_body(self):
        """Get new body."""
        self.update()
        return self._skeleton

    def log_times(self):
        """Log times."""
        if not self._timed:
            return
        if len(self._capture_times) > 0:
            logging.info(
                f"Average capture time (ms): {np.mean(self._capture_times) / 1e6}"
            )
        if len(self._tracking_times) > 0:
            logging.info(
                f"Average body frame time (ms): {np.mean(self._tracking_times) / 1e6}"
            )
