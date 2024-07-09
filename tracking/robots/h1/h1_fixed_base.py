"""Fixed H1 Control State."""
import numpy as np

from tracking import BodyTracker
from .h1_fixed_base_state import H1FixedBaseState


PYKINECT_JOINTS = [
    "pelvis",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
]


class H1FixedBase:
    """Control State for H1 with fixed legs."""

    def __init__(self, tracker: BodyTracker):
        """Initialize H1 Control State."""
        self._state = None
        self._ref_state = None
        self._tracker = tracker

    def update(self):
        """Update H1 Control State."""
        self._tracker.update()
        if self._state is None:
            self._ref_state = H1FixedBaseState()
            self._ref_state.update_from_transforms(**self._tracker.reference_skeleton)
            self._state = H1FixedBaseState()
        self._state.update_from_transforms(**self._tracker.skeleton)

    def get_action(self):
        """Get bounded H1 Control Action with reference."""
        return self._angles_to_action(self._state.get_joint_angles(self._ref_state))

    def _angles_to_action(self, angles):
        """Convert angles to action."""
        return np.array(
            [
                angles["pelvis"]["x"],
                angles["pelvis"]["y"],
                angles["pelvis"]["rotation"],
                angles["left_shoulder"]["pitch"],
                angles["left_shoulder"]["roll"],
                angles["left_shoulder"]["yaw"],
                angles["left_elbow"]["angle"],
                angles["right_shoulder"]["pitch"],
                angles["right_shoulder"]["roll"],
                angles["right_shoulder"]["yaw"],
                angles["right_elbow"]["angle"],
                0,
                0,
            ]
        )

    @property
    def joint_angles(self):
        """Get joint angles."""
        joint_angles = self.get_action()
        return {
            "pelvis": np.array(joint_angles[:3]),
            "left_shoulder": np.array(joint_angles[3:6]),
            "left_elbow": np.array(joint_angles[6]),
            "right_shoulder": np.array(joint_angles[7:10]),
            "right_elbow": np.array(joint_angles[10]),
        }
