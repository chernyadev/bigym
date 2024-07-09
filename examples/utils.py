"""Utils for example scripts."""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import string


class Window:
    """Class to hold a window for showing images."""

    def __init__(
        self,
        name: str = None,
        width: int = 800,
        height: int = 800,
        x: int = 0,
        y: int = 0,
    ):
        """Initialize."""
        if name is not None:
            self.name = name
        else:
            self.name = "".join(
                random.choices(string.ascii_uppercase + string.digits, k=10)
            )
        self.width = width
        self.height = height
        self.x = x
        self.y = y

        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.name, self.width, self.height)
        cv2.moveWindow(self.name, self.x, self.y)

    def show(self, frame):
        """Show frame. Don't forget to call cv2.waitKey(1)."""
        cv2.imshow(self.name, frame)

    def close(self):
        """Close window."""
        cv2.destroyWindow(self.name)


def set_other_joints_to_zero(action, indexes, value=None):
    """Set other joints to zero."""
    new_action = np.zeros_like(action)
    for i in indexes:
        if value is None:
            new_action[i] = action[i]
        else:
            new_action[i] = value
    return new_action


def init_subplots(n):
    """Initialize subplots."""
    fig = plt.figure(figsize=(8, 2 * n))
    axs = [fig.add_subplot(n, 1, i + 1) for i in range(n)]
    return fig, axs


def update_plots(axs, requested, expected, actual, xlim=None, ylim=None):
    """Update plots."""
    for i, (r, e, a) in enumerate(zip(requested, expected, actual)):
        axs[i].clear()  # Clear previous plot
        axs[i].plot(r, label="Request")
        axs[i].plot(e, label="Expected")
        axs[i].plot(a, label="Actual")
        axs[i].set_title(f"Variable {i}")
        axs[i].legend()
        if xlim is not None:
            axs[i].set_xlim(xlim)
        if ylim is not None:
            axs[i].set_ylim(ylim)

    plt.tight_layout()
