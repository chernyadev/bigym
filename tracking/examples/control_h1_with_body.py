"""An example of controlling the H1 robot with a Kinect."""
import numpy as np
import cv2
import argparse

from bigym.envs.manipulation import StackBlocks
from bigym.action_modes import JointPositionActionMode

from tracking.robots import H1FixedBase
from tracking.kinect import KinectTracker
from tracking.mediapipe import MediapipeTracker

from examples.utils import Window, set_other_joints_to_zero


# change printing format to 2 d.p. with +/- sign and longer wrap width
np.set_printoptions(precision=2, floatmode="fixed", suppress=True, linewidth=2000)

parser = argparse.ArgumentParser()
parser.add_argument("--tracker", type=str, default="kinect")
parser.add_argument("--base_fixed", action="store_true")
parser.add_argument("--zero_action", action="store_true")
parser.add_argument("--only", type=str, default=None)
parser.add_argument("--arm", type=str, default=None)
parser.add_argument("--value", type=float, default=None)
parser.add_argument("--camera", type=str, default="head")
args = parser.parse_args()

base_fixed = args.base_fixed
zero_action = args.zero_action
only = args.only
arm = args.arm
camera = args.camera
cam_key = f"rgb_{camera}"
value = args.value
tracker_type = args.tracker

kinect_window = Window()
mujoco_window = Window(x=800)

if tracker_type == "kinect":
    tracker = KinectTracker(timed=True)
elif tracker_type == "mediapipe":
    tracker = MediapipeTracker(timed=True)

h1 = H1FixedBase(tracker)
env = StackBlocks(
    action_mode=JointPositionActionMode(
        absolute=True,
        block_until_reached=False,
        floating_base=True,
    ),
    cameras=[camera],
    camera_resolution=480,
    render_mode=None,
)

env.reset()

episode_length = 5000
i = 0
j = 0

while True:
    position = env.robot.qpos_actuated
    h1.update()
    action = h1.get_action()

    if base_fixed:
        action[:3] = np.zeros(3)

    if zero_action:
        indexes = []
    else:
        if only == "shoulder_pitch":
            indexes = [3, 7]
        elif only == "shoulder_roll":
            indexes = [4, 8]
        elif only == "shoulder_yaw":
            indexes = [5, 9]
        elif only == "elbow":
            indexes = [6, 10]
        else:
            indexes = [i for i in range(len(action))]

    if arm == "left":
        indexes = indexes[:1]
    elif arm == "right":
        indexes = indexes[1:]

    action[:3] = np.zeros(3)

    action = set_other_joints_to_zero(action, indexes, value=value)
    obs, reward, terminated, truncated, info = env.step(action)

    if i % 10 == 0:
        position = env.robot.qpos_actuated
        print(f"p: {position}")
        print(f"a: {action}")

    mujoco_window.show(obs[cam_key].transpose([1, 2, 0]))
    kinect_window.show(tracker.image)
    if cv2.waitKey(1) == ord("q"):
        break
    if i % episode_length == 0:
        j += 1
        env.reset()
    i += 1
env.close()

tracker.log_times()
