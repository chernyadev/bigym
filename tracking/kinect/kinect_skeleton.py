"""Fixed H1 Control State."""
from tracking.utils import Transform
from tracking import Skeleton

from pykinect_azure.k4abt.body import Body


PYKINECT_JOINTS = [
    "pelvis",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
]


class KinectSkeleton(Skeleton):
    """Create Skeleton from Kinect body."""

    def update(self, body: Body):
        """Update skeleton."""
        if not self._joint_ref_tfs:
            for joint in body.joints:
                name = joint.name.replace(" ", "_")
                if name in PYKINECT_JOINTS:
                    self._joint_ref_tfs[name] = Transform.from_joint(joint)
            self._ref_skeleton = self._create_skeleton(body, self._joint_ref_tfs)
        self._skeleton = self._create_skeleton(body, self._joint_ref_tfs)

    def _create_skeleton(self, body: Body, ref_tfs: dict):
        """Create skeleton."""
        skeleton = {}
        body = {joint.name.replace(" ", "_"): joint for joint in body.joints}
        skeleton["pelvis"] = Transform.from_joint(
            body["pelvis"],
            reference_transform=ref_tfs["pelvis"],
        )
        skeleton["left_shoulder"] = Transform.from_joint(
            body["left_shoulder"],
            reference_transform=ref_tfs["left_shoulder"],
        )
        skeleton["left_elbow"] = Transform.from_joint(
            body["left_elbow"],
            reference_transform=ref_tfs["left_shoulder"],
        )
        skeleton["left_wrist"] = Transform.from_joint(
            body["left_wrist"],
            reference_transform=ref_tfs["left_elbow"],
        )
        skeleton["right_shoulder"] = Transform.from_joint(
            body["right_shoulder"],
            reference_transform=ref_tfs["right_shoulder"],
        )
        skeleton["right_elbow"] = Transform.from_joint(
            body["right_elbow"],
            reference_transform=ref_tfs["right_shoulder"],
        )
        skeleton["right_wrist"] = Transform.from_joint(
            body["right_wrist"],
            reference_transform=ref_tfs["right_elbow"],
        )
        return skeleton
