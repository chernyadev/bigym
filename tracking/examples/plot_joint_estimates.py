"""An example of plotting joint estimates from Kinect."""
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Tuple

from tracking.utils import Transform
from tracking.robots import H1FixedBase
from tracking.kinect import KinectTracker
from tracking.mediapipe import MediapipeTracker

from examples.utils import Window


JOINTS = [
    "pelvis",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
]

# change printing format to 2 d.p. with +/- sign and longer wrap width
np.set_printoptions(precision=2, floatmode="fixed", suppress=True, linewidth=2000)


def update_stickman(transforms: Tuple[Transform], color="red"):
    """Update stickman's positions."""
    positions = [t.position for t in transforms]
    (p, ls, le, lw, rs, re, rw) = positions
    # draw lines from pelvis to left shoulder, elbow, wrist
    plt.plot([p.x, ls.x], [p.y, ls.y], [p.z, ls.z], color=color)
    plt.plot([ls.x, le.x], [ls.y, le.y], [ls.z, le.z], color=color)
    plt.plot([le.x, lw.x], [le.y, lw.y], [le.z, lw.z], color=color)
    # draw lines from pelvis to right shoulder, elbow, wrist
    plt.plot([p.x, rs.x], [p.y, rs.y], [p.z, rs.z], color=color)
    plt.plot([rs.x, re.x], [rs.y, re.y], [rs.z, re.z], color=color)
    plt.plot([re.x, rw.x], [re.y, rw.y], [re.z, rw.z], color=color)
    # draw line from left shoulder to right shoulder
    plt.plot([ls.x, rs.x], [ls.y, rs.y], [ls.z, rs.z], color=color)

    x_lim = [p.x for p in positions]
    y_lim = [p.y for p in positions]
    z_lim = [p.z for p in positions]

    plt.draw()

    return (
        (min(x_lim), max(x_lim)),
        (min(y_lim), max(y_lim)),
        (min(z_lim), max(z_lim)),
    )


def update_labels(angles: dict, transforms: dict, color="red", use_degrees=True):
    """Update plot labels."""
    for joint, angle in angles.items():
        tf = transforms[joint]
        if angle.ndim == 0:
            angle = np.array([angle])
        if use_degrees:
            angle = [180 * rad / np.pi for rad in angle]
        text = f"{[int(round(i)) for i in angle]}"
        ax.text(tf.position.x, tf.position.y, tf.position.z, text, color=color)


def update_positions(transforms: dict, color="red"):
    """Update plot position labels."""
    for joint, tf in transforms.items():
        p = tf.position.normalized
        text = f"[x: {p.x:.1f}, y: {p.y:.1f}, z: {p.z:.1f}]"
        ax.text(tf.position.x, tf.position.y, tf.position.z, text, color=color)


parser = argparse.ArgumentParser()
parser.add_argument("--tracker", type=str, default="kinect")
parser.add_argument("--tracker_window", action="store_true", default=False)
args = parser.parse_args()

tracker_type = args.tracker
show_window = args.tracker_window

if show_window:
    window = Window(x=800)

if tracker_type == "kinect":
    tracker = KinectTracker(timed=True)
elif tracker_type == "mediapipe":
    tracker = MediapipeTracker(timed=True)

h1 = H1FixedBase(tracker)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
lines = []
plt.show(block=False)

x_min, x_max = -1e-4, 1e-4
y_min, y_max = -1e-4, 1e-4
z_min, z_max = -1e-4, 1e-4

while True:
    h1.update()
    action = h1.get_action()

    tfs = tuple(tracker.skeleton.values())
    rtfs = tuple(tracker.reference_skeleton.values())

    angles = h1.joint_angles
    action_dict = angles.copy()
    angles["pelvis"] = action[:3]
    angles["left_shoulder"] = action[3:6]
    angles["left_elbow"] = action[6]
    angles["right_shoulder"] = action[7:10]
    angles["right_elbow"] = action[10]

    ax.cla()
    x, y, z = update_stickman(tfs)
    x2, y2, z2 = update_stickman(rtfs, color="blue")
    update_labels(action_dict, tracker.skeleton)
    j_tfs = tracker.skeleton
    labels = dict([(joint, tf) for joint, tf in j_tfs.items() if "wrist" in joint])
    update_positions(labels)

    x_min = min(x_min, *x)
    x_max = max(x_max, *x)
    ax.set_xlim(x_min, x_max)

    y_min = min(y_min, *y)
    y_max = max(y_max, *y)
    ax.set_ylim(y_min, y_max)

    z_min = min(z_min, *z)
    z_max = max(z_max, *z)
    ax.set_zlim(z_min, z_max)

    plt.pause(0.001)

    if show_window:
        window.show(tracker.image)
        cv2.waitKey(1)
