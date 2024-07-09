"""Body tracking using mediapipe."""
import cv2
import logging
import numpy as np
import pykinect_azure as pykinect
import time
from enum import Enum

import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

from tracking import BodyTracker
from tracking.mediapipe import MediapipeSkeleton


class MediapipeTrackingModel(Enum):
    """Mediapipe tracking models."""

    LITE = "lite"
    FULL = "full"
    HEAVY = "heavy"

    def path(self):
        """Get model path."""
        return f"tracking/models/pose_landmarker_{self.value}.task"


class MediapipeTracker(BodyTracker):
    """Body tracking using mediapipe."""

    def __init__(
        self,
        model: MediapipeTrackingModel = MediapipeTrackingModel.HEAVY,
        *args,
        **qwargs,
    ):
        """Initialize body tracker."""
        super().__init__(*args, **qwargs)

        pykinect.initialize_libraries()
        device_config = pykinect.default_configuration
        device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
        # device_config.depth_mode = pykinect.K4A_DEPTH_MODE_PASSIVE_IR
        device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
        device_config.create
        self._device = pykinect.start_device(config=device_config)

        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model.path()),
            running_mode=VisionRunningMode.IMAGE,
            num_poses=1,
        )

        self._landmarker = PoseLandmarker.create_from_options(options)
        self._last_capture = None
        self._last_body_frame = None
        self._skeleton = MediapipeSkeleton()

    def update(self):
        """Update body tracker."""
        self._last_result = self._track()
        self._skeleton.update(self._last_result)

    @property
    def image(self):
        """Draws the landmarks and the connections on the image."""
        image = self._last_image
        pose_landmarks_list = self._last_result.pose_landmarks
        annotated_image = np.copy(image)

        # Loop through the detected poses to visualize.
        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]

            # Draw the pose landmarks.
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend(
                [
                    landmark_pb2.NormalizedLandmark(
                        x=landmark.x, y=landmark.y, z=landmark.z
                    )
                    for landmark in pose_landmarks
                ]
            )
            try:
                solutions.drawing_utils.draw_landmarks(
                    annotated_image,
                    pose_landmarks_proto,
                    solutions.pose.POSE_CONNECTIONS,
                    solutions.drawing_styles.get_default_pose_landmarks_style(),
                )
            except Exception as e:
                logging.error(e)
                return annotated_image
        return annotated_image

    def _track(self):
        """Use mediapipe to track the body."""
        color_image, depth_image = self._wait_for_images()
        image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        image = mp.Image(data=image, image_format=mp.ImageFormat.SRGB)
        if self._timed:
            start = time.time_ns()
            result = self._landmarker.detect(image)
            end = time.time_ns()
            self._tracking_times.append(end - start)
        else:
            result = self._landmarker.detect(image)
        self._last_image = cv2.cvtColor(image.numpy_view(), cv2.COLOR_RGB2BGR)
        return result

    def _capture(self):
        """Capture image."""
        if self._timed:
            start = time.time_ns()
            self._last_capture = self._device.update()
            end = time.time_ns()
            self._capture_times.append(end - start)
        else:
            self._last_capture = self._device.update()
        return self._last_capture

    def _wait_for_images(self):
        """Wait for images."""
        while True:
            capture = self._capture()
            ret, color_image = capture.get_color_image()
            dep_ret, depth_image = capture.get_depth_image()
            if not ret or not dep_ret:
                logging.error(f"Failed to get images, ret={ret}, dep_ret={dep_ret}")
                continue
            break
        return color_image, depth_image
