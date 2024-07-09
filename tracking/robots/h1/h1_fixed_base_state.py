"""Class to hold state of H1 robot."""
from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np

from tracking.utils import Transform, Vector3


class JointStatesBase(ABC):
    """Joint State Base."""

    @abstractmethod
    def set(self):
        """Set state."""
        pass

    @abstractmethod
    def set_from_transforms(self):
        """Set state from transforms."""
        pass


class H1BaseState(JointStatesBase):
    """Class to hold fixed base state of H1 robot."""

    def __init__(self, x, y, rotation):
        """Initialize."""
        self.x = x
        self.y = y
        self.rotation = rotation

    def set(self, x, y, rotation):
        """Set base."""
        self.x = x
        self.y = y
        self.rotation = rotation

    def set_from_transforms(self, pelvis: Transform):
        """Set base from pelvis transform."""
        self.x = pelvis.position[2]
        self.y = pelvis.position[1]
        self.rotation = pelvis.euler[0]


class H1TorsoState(JointStatesBase):
    """Class to hold torso state of H1 robot."""

    def __init__(self, rotation):
        """Initialize."""
        self.rotation = rotation

    def set(self, rotation):
        """Set rotation."""
        self.rotation = rotation

    def set_from_transforms(
        self,
        pelvis: Transform,
        left_shoulder: Transform,
        right_shoulder: Transform,
    ):
        """Set base from transforms."""
        v1 = right_shoulder.vector_to(left_shoulder)
        # todo: check if this is correct
        v2 = Vector3(pelvis.euler)
        self.rotation = v1.angle_to(v2)


class H1ShoulderState(JointStatesBase):
    """Class to hold shoulder state of H1 robot."""

    def __init__(self, pitch, roll, yaw, flip=False):
        """Initialize."""
        self.pitch = pitch
        self.roll = roll
        self.yaw = yaw
        self._flip = flip

    def set(self, pitch, roll, yaw):
        """Set shoulder."""
        self.pitch = pitch
        self.roll = roll
        self.yaw = yaw

    def set_from_transforms(
        self, shoulder: Transform, elbow: Transform, wrist: Transform, ref: Transform
    ):
        """Set shoulder from transforms."""
        upper_arm = shoulder.vector_to(elbow)
        lower_arm = elbow.vector_to(wrist)
        n = upper_arm.normalized

        self.pitch = np.arcsin(n.z)
        self.roll = np.arcsin(-n.y) + np.pi / 2
        self.yaw = lower_arm.angle_to(shoulder.z_axis)

        # Needs to be flipped if right arm
        if self._flip:
            self.roll = -self.roll


class H1ArmState:
    """Class to hold arm state of H1 robot."""

    def __init__(self, shoulder: H1ShoulderState, elbow):
        """Initialize."""
        self.shoulder = shoulder
        self.elbow = elbow

    def set_arm(self, shoulder: H1ShoulderState, elbow):
        """Set arm."""
        self.shoulder = shoulder
        self.elbow = elbow

    def set_arm_from_transforms(
        self,
        shoulder: Transform,
        elbow: Transform,
        wrist: Transform,
        pelvis: Transform,
    ):
        """Set arm from transforms."""
        self.shoulder.set_from_transforms(shoulder, elbow, wrist, pelvis)
        v1 = shoulder.vector_to(elbow)
        v2 = elbow.vector_to(wrist)
        self.elbow = -v1.angle_to(v2) + np.pi / 2


JOINT_RANGES = {
    "left_shoulder_pitch": (-2.87, 2.87),
    "left_shoulder_roll": (-0.34, 3.11),
    "left_shoulder_yaw": (-1.3, 4.45),
    "left_elbow": (-1.25, 2.61),
    "right_shoulder_pitch": (-2.87, 2.87),
    "right_shoulder_roll": (-3.11, 0.34),
    "right_shoulder_yaw": (-4.45, 1.3),
    "right_elbow": (-1.25, 2.61),
}


class H1FixedBaseState:
    """Class to hold joint state of H1 robot."""

    def __init__(self):
        """Initialize."""
        self.base = H1BaseState(0, 0, 0)
        self.left_arm = H1ArmState(H1ShoulderState(0, 0, 0), 0)
        self.right_arm = H1ArmState(H1ShoulderState(0, 0, 0, flip=True), 0)
        self.torso = H1TorsoState(0)

    def set_state(
        self,
        base: H1BaseState,
        left_arm: H1ArmState,
        right_arm: H1ArmState,
        torso: H1TorsoState,
    ):
        """Set state."""
        self.base = base
        self.left_arm = left_arm
        self.right_arm = right_arm
        self.torso = torso

    def update_from_transforms(
        self,
        pelvis: Transform,
        left_shoulder: Transform,
        left_elbow: Transform,
        left_wrist: Transform,
        right_shoulder: Transform,
        right_elbow: Transform,
        right_wrist: Transform,
    ):
        """Update from transforms."""
        self.base.set_from_transforms(pelvis)
        self.left_arm.set_arm_from_transforms(
            left_shoulder,
            left_elbow,
            left_wrist,
            pelvis,
        )
        self.right_arm.set_arm_from_transforms(
            right_shoulder,
            right_elbow,
            right_wrist,
            pelvis,
        )
        self.torso.set_from_transforms(
            pelvis,
            left_shoulder,
            right_shoulder,
        )

    def get_joint_angles(self, ref: H1FixedBaseState):
        """Get joint angles with reference."""
        return {
            "pelvis": {
                "x": self.base.x - ref.base.x,
                "y": self.base.y - ref.base.y,
                "rotation": self.base.rotation - ref.base.rotation,
            },
            "left_shoulder": {
                "pitch": np.clip(
                    self.left_arm.shoulder.pitch - ref.left_arm.shoulder.pitch,
                    *JOINT_RANGES["left_shoulder_pitch"],
                ),
                "roll": np.clip(
                    self.left_arm.shoulder.roll - ref.left_arm.shoulder.roll,
                    *JOINT_RANGES["left_shoulder_roll"],
                ),
                "yaw": np.clip(
                    self.left_arm.shoulder.yaw - ref.left_arm.shoulder.yaw,
                    *JOINT_RANGES["left_shoulder_yaw"],
                ),
            },
            "left_elbow": {
                "angle": np.clip(
                    self.left_arm.elbow - ref.left_arm.elbow,
                    *JOINT_RANGES["left_elbow"],
                ),
            },
            "right_shoulder": {
                "pitch": np.clip(
                    self.right_arm.shoulder.pitch - ref.right_arm.shoulder.pitch,
                    *JOINT_RANGES["right_shoulder_pitch"],
                ),
                "roll": np.clip(
                    self.right_arm.shoulder.roll - ref.right_arm.shoulder.roll,
                    *JOINT_RANGES["right_shoulder_roll"],
                ),
                "yaw": np.clip(
                    self.right_arm.shoulder.yaw - ref.right_arm.shoulder.yaw,
                    *JOINT_RANGES["right_shoulder_yaw"],
                ),
            },
            "right_elbow": {
                "angle": np.clip(
                    self.right_arm.elbow - ref.right_arm.elbow,
                    *JOINT_RANGES["right_elbow"],
                ),
            },
            "torso": {
                "rotation": self.torso.rotation - ref.torso.rotation,
            },
        }
