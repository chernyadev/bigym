"""An example to print joint estimates from Kinect."""
import argparse
import numpy as np
import cv2

from tracking.robots import H1FixedBase
from tracking.kinect import KinectTracker
from tracking.mediapipe import MediapipeTracker

from examples.utils import Window


# change printing format to 2 d.p. with +/- sign and longer wrap width
np.set_printoptions(precision=2, floatmode="fixed", suppress=True, linewidth=1000)

parser = argparse.ArgumentParser()
parser.add_argument("--tracker", type=str, default="kinect")
parser.add_argument("--tracker_window", action="store_true", default=False)
args = parser.parse_args()

tracker_type = args.tracker
show_window = args.tracker_window

if show_window:
    window = Window()

if tracker_type == "kinect":
    tracker = KinectTracker(timed=True)
elif tracker_type == "mediapipe":
    tracker = MediapipeTracker(timed=True)

h1 = H1FixedBase(tracker)

h1.update()
initial_action = h1.get_action().copy()

i = 0

while True:
    h1.update()
    action = h1.get_action()
    delta_action = action - initial_action

    if i % 10 == 0:
        print(f"a: {action}")
        print(f"d: {delta_action}")

    if show_window:
        window.show(tracker.image)

    if cv2.waitKey(1) == ord("q"):
        break

    i += 1

tracker.log_times()
