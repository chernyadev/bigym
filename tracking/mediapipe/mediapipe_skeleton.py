"""Creates skeleton from mediapipe landmark result."""
import logging
import numpy as np

from tracking.utils import Transform, Vector3
from tracking import Skeleton

from mediapipe.tasks.python.vision import PoseLandmarkerResult
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmark


MEDIAPIPE_JOINTS = {
    "left_hip": {
        "name": "left_hip",
        "index": 23,
        "include": False,
    },
    "left_shoulder": {
        "name": "left_shoulder",
        "index": 11,
        "include": True,
    },
    "left_elbow": {
        "name": "left_elbow",
        "index": 13,
        "include": True,
    },
    "left_wrist": {
        "name": "left_wrist",
        "index": 15,
        "include": True,
    },
    "right_hip": {
        "name": "right_hip",
        "index": 24,
        "include": False,
    },
    "right_shoulder": {
        "name": "right_shoulder",
        "index": 12,
        "include": True,
    },
    "right_elbow": {
        "name": "right_elbow",
        "index": 14,
        "include": True,
    },
    "right_wrist": {
        "name": "right_wrist",
        "index": 16,
        "include": True,
    },
    "pelvis": {
        "name": "pelvis",
        "index": 33,
        "include": True,
    },
}


class MediapipeSkeleton(Skeleton):
    """Creates skeleton from mediapipe landmark result.

    ------------------------------------------------
    | WARNING: The math in this class is not fully  |
    | tested and may not work as expected. Work is  |
    | currently paused due to instability from the  |
    | mediapipe tracking.                           |
    ------------------------------------------------
    """

    def update(self, landmark_result: PoseLandmarkerResult):
        """Update skeleton."""
        if not self._joint_ref_tfs:
            _, tfs = self._included_landmarks(landmark_result)
            if not tfs:
                logging.warning("No body detected, unable to set reference skeleton.")
                return
            self._joint_ref_tfs = tfs
            self._ref_skeleton = self._create_skeleton(
                landmark_result, self._joint_ref_tfs
            )
        self._skeleton = self._create_skeleton(landmark_result, self._joint_ref_tfs)

    def _create_skeleton(self, landmark_result: PoseLandmarkerResult, ref_tfs: dict):
        """Create skeleton."""
        lms, tfs = self._included_landmarks(landmark_result)
        if not lms or not tfs:
            logging.warning("No body detected, using previous skeleton.")
            return self._skeleton
        skeleton = {}
        skeleton["pelvis"] = Transform.from_landmark(
            lms["pelvis"],
            reference_transform=ref_tfs["pelvis"],
        )
        skeleton["left_shoulder"] = Transform.from_landmark(
            lms["left_shoulder"],
            reference_transform=ref_tfs["left_shoulder"],
        )
        skeleton["left_elbow"] = Transform.from_landmark(
            lms["left_elbow"],
            reference_transform=ref_tfs["left_shoulder"],
        )
        skeleton["left_wrist"] = Transform.from_landmark(
            lms["left_wrist"],
            reference_transform=ref_tfs["left_elbow"],
        )
        skeleton["right_shoulder"] = Transform.from_landmark(
            lms["right_shoulder"],
            reference_transform=ref_tfs["right_shoulder"],
        )
        skeleton["right_elbow"] = Transform.from_landmark(
            lms["right_elbow"],
            reference_transform=ref_tfs["right_shoulder"],
        )
        skeleton["right_wrist"] = Transform.from_landmark(
            lms["right_wrist"],
            reference_transform=ref_tfs["right_elbow"],
        )
        return skeleton

    def _landmark_to_numpy(self, landmark):
        """Convert landmark to numpy array."""
        return np.array(
            [landmark.x, landmark.y, landmark.z, landmark.visibility, landmark.presence]
        )

    def _included_landmarks(self, landmark_result: PoseLandmarkerResult):
        """Get transforms from landmark result."""
        # Using first skeleton detected
        if len(landmark_result.pose_landmarks) == 0:
            logging.info("No pose landmarks detected")
            return None, None
        landmarks = landmark_result.pose_landmarks[0]

        # Create reference transform as mediapipe does not provide any.
        right_shoulder = self._landmark_to_numpy(
            landmarks[MEDIAPIPE_JOINTS["right_shoulder"]["index"]]
        )
        left_shoulder = self._landmark_to_numpy(
            landmarks[MEDIAPIPE_JOINTS["left_shoulder"]["index"]]
        )
        right_hip = self._landmark_to_numpy(
            landmarks[MEDIAPIPE_JOINTS["right_hip"]["index"]]
        )
        left_hip = self._landmark_to_numpy(
            landmarks[MEDIAPIPE_JOINTS["left_hip"]["index"]]
        )
        pelvis = (right_hip + left_hip) / 2
        ref = Transform.from_three_positions(
            Vector3(right_shoulder[:3]),
            Vector3(left_shoulder[:3]),
            Vector3(pelvis[:3]),
        )

        pelvis = NormalizedLandmark(
            x=pelvis[0],
            y=pelvis[1],
            z=pelvis[2],
            visibility=pelvis[3],
            presence=pelvis[4],
        )
        landmarks.append(pelvis)

        indexes_and_names = [
            (j["index"], j["name"]) for j in MEDIAPIPE_JOINTS.values() if j["include"]
        ]
        landmarks = {
            name: self._landmark_to_numpy(landmarks[index])
            for index, name in indexes_and_names
        }
        ref_tfs = {
            name: Transform.from_position_quaternion(
                landmarks[name][:3], ref.quaternion
            )
            for name, lm in landmarks.items()
        }
        transforms = {
            name: Transform.from_landmark(lm, reference_transform=ref_tfs[name])
            for name, lm in landmarks.items()
        }
        return landmarks, transforms
