"""A 4x4 matrix representing a reference transform."""
from __future__ import annotations
import logging
import numpy as np

from tracking.utils import Vector3

from scipy.spatial.transform import Rotation
from pykinect_azure.k4abt.joint import Joint


LANDMARK_THRESHOLD = 0.7


class Transform:
    """A 4x4 matrix representing a reference transform."""

    def __init__(self, matrix: np.ndarray = np.eye(4)):
        """Initialize a transform with origin and basis vectors."""
        assert matrix.shape == (4, 4), "Matrix must be 4x4"
        self.matrix = matrix

    @staticmethod
    def from_position_quaternion(position: np.ndarray, quaternion: np.ndarray):
        """Create a transform from a position and quaternion.

        Args:
            position: A position.
            quaternion: A quaternion [x, y, z, w].
        """
        assert position.shape == (3,), "Position must be of shape (3,)"
        assert quaternion.shape == (4,), "Quaternion must be of shape (4,)"
        return Transform(
            Transform._matrix_from_position_rotation_matrix(
                position, Rotation.from_quat(quaternion).as_matrix()
            )
        )

    @staticmethod
    def from_position_euler(position: np.ndarray, euler: np.ndarray):
        """Create a transform from a position and euler angles."""
        assert position.shape == (3,), "Position must be of shape (3,)"
        assert euler.shape == (3,), "Euler angles must be of shape (3,)"
        return Transform(
            Transform._matrix_from_position_rotation_matrix(
                position, Rotation.from_euler("xyz", euler).as_matrix()
            )
        )

    @staticmethod
    def from_joint(joint: Joint, reference_transform: Transform = None):
        """Create a transform from a joint."""
        ref = (
            Transform(np.eye(4)) if reference_transform is None else reference_transform
        )
        p, o = joint.position, joint.orientation
        p, o = Vector3([p.x, p.y, p.z]), np.array([o.x, o.y, o.z, o.w])
        tf = Transform.from_position_quaternion(p, o)
        tf = tf.transform(ref)
        tf.position = tf.position * 1e-5
        return tf

    @staticmethod
    def from_landmark(landmark: np.ndarray, reference_transform: Transform = None):
        """Create a transform from a landmark.

        Args:
            landmark: A landmark [x, y, z, visibility, presence].
            reference_transform: A reference transform.
        """
        assert len(landmark) == 5, f"Landmark must length 5, not {len(landmark)}"
        if landmark[3] < LANDMARK_THRESHOLD or landmark[4] < LANDMARK_THRESHOLD:
            logging.warning("Your head, hands and hips must be visible")
        ref = (
            Transform(np.eye(4)) if reference_transform is None else reference_transform
        )
        p = Vector3(landmark[:3])
        o = ref.quaternion
        tf = Transform.from_position_quaternion(p, o)
        tf = tf.transform(ref)
        return tf

    @staticmethod
    def from_transforms(tf1: Transform, tf2: Transform, tf3: Transform):
        """Set the transform from two transforms."""
        p = tf1.position
        v1 = tf1.vector_to(tf2).normalized
        v2 = tf1.vector_to(tf3).normalized
        n1 = v1.cross(v2).normalized
        n2 = v1.cross(n1).normalized
        return Transform(
            Transform._matrix_from_position_rotation_matrix(
                p, np.column_stack([-n2.as_ndarray(), n1.as_ndarray(), v1.as_ndarray()])
            )
        )

    @staticmethod
    def from_three_positions(p1: Vector3, p2: Vector3, p3: Vector3):
        """Creates transform from three positions.

        The transform is created such that the z axis points from p1 to p2,
        the y axis is the cross product of the vector from p1 to p3 and the
        vector from p1 to p2, and the x axis is the cross product of the y
        axis and the z axis.

        The three points lie on the x-z plane and appear in a counter-clockwise
        order when looking down the y axis.
        """
        v1 = p1.vector_to(p2).normalized
        v2 = p1.vector_to(p3).normalized
        n1 = v2.cross(v1).normalized
        n2 = n1.cross(v1).normalized
        return Transform(
            Transform._matrix_from_position_rotation_matrix(
                p1, np.column_stack([n2.as_ndarray(), n1.as_ndarray(), p1.as_ndarray()])
            )
        )

    @staticmethod
    def _matrix_from_position_rotation_matrix(
        position: np.ndarray, rotation_matrix: np.ndarray
    ):
        """Create a matrix from a position and rotation matrix."""
        assert position.shape == (3,), "Position must be of shape (3,)"
        assert rotation_matrix.shape == (
            3,
            3,
        ), "Rotation matrix must be of shape (3, 3)"
        matrix = np.eye(4)
        matrix[:3, 3] = position
        matrix[:3, :3] = rotation_matrix
        return matrix

    @property
    def position(self):
        """Return the position of the transform."""
        return Vector3(self.matrix[:3, 3])

    @position.setter
    def position(self, value):
        """Set the position of the transform."""
        assert value.shape == (3,), f"Position must be of shape (3,) not {value.shape}"
        self.matrix[:3, 3] = value

    @property
    def rotation_matrix(self):
        """Return the orientation of the transform."""
        return self.matrix[:3, :3]

    @property
    def quaternion(self):
        """Return the orientation of the transform as a quaternion."""
        return Rotation.from_matrix(self.rotation_matrix).as_quat()

    @property
    def euler(self):
        """Return the orientation of the transform as euler angles."""
        return Rotation.from_matrix(self.rotation_matrix).as_euler("xyz")

    @property
    def x_axis(self):
        """Return the normalized x axis of the transform."""
        return Vector3(self.rotation_matrix[:, 0]).normalized

    @property
    def y_axis(self):
        """Return the normalized y axis of the transform."""
        return Vector3(self.rotation_matrix[:, 1]).normalized

    @property
    def z_axis(self):
        """Return the normalized z axis of the transform."""
        return Vector3(self.rotation_matrix[:, 2]).normalized

    def vector_to(self, other_transform: Transform) -> Vector3:
        """Return the vector from this transform to another transform."""
        return other_transform.position - self.position

    def transform(self, other_transform: Transform) -> Transform:
        """Transform another transform by this transform."""
        return Transform(self.matrix @ other_transform.matrix)
