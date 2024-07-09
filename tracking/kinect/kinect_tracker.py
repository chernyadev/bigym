"""Body tracking using Kinect Azure."""
import cv2
import pykinect_azure as pykinect
import time

from tracking import BodyTracker
from tracking.kinect import KinectSkeleton


class KinectTracker(BodyTracker):
    """Body tracking using Kinect Azure."""

    def __init__(self, *args, **qwargs):
        """Initialize body tracker."""
        super().__init__(*args, **qwargs)
        pykinect.initialize_libraries(track_body=True)
        device_config = pykinect.default_configuration
        device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_OFF
        device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
        self._device = pykinect.start_device(config=device_config)
        self._tracker = pykinect.start_body_tracker(
            model_type=pykinect.K4ABT_LITE_MODEL
        )
        self._last_capture = None
        self._last_body_frame = None
        self._skeleton = KinectSkeleton()

    def update(self):
        """Update body tracker."""
        if self._timed:
            start = time.time_ns()
            self._last_capture = self._device.update()
            capture_end = time.time_ns()
            self._last_body_frame = self._tracker.update()
            body = self._last_body_frame.get_body(0)
            tracking_end = time.time_ns()
            self._capture_times.append(capture_end - start)
            self._tracking_times.append(tracking_end - capture_end)
        else:
            self._last_capture = self._device.update()
            self._last_body_frame = self._tracker.update()
            body = self._last_body_frame.get_body(0)
        self._skeleton.update(body)

    @property
    def image(self):
        """Get image."""
        ret_color, depth_color_image = self._last_capture.get_colored_depth_image()
        ret_depth, body_image_color = self._last_body_frame.get_segmentation_image()
        if not ret_color or not ret_depth:
            return None
        combined_image = cv2.addWeighted(
            depth_color_image, 0.6, body_image_color, 0.4, 0
        )
        combined_image = self._last_body_frame.draw_bodies(combined_image)
        return combined_image
