"""A class to represent a 3D vector."""
import numpy as np


class Vector3(np.ndarray):
    """A class to represent a 3D vector."""

    def __new__(cls, a, dtype=np.float64, order=None):
        """Create a vector."""
        assert len(a) == 3, f"Input array must have a length of 3, not {len(a)}"
        obj = np.asarray(a, dtype, order).view(cls)
        return obj

    def __array_wrap__(self, out_arr, context=None):
        """Wrap the array."""
        return np.ndarray.__array_wrap__(self, out_arr, context)

    def __array_finalize__(self, obj):
        """Finalize the array."""
        if obj is None:
            return
        if obj.shape != (3,):
            raise ValueError(f"Input array must have a shape of (3,), not {obj.shape}")

    @property
    def x(self):
        """Return the x component of the vector."""
        return self[0]

    @property
    def y(self):
        """Return the y component of the vector."""
        return self[1]

    @property
    def z(self):
        """Return the z component of the vector."""
        return self[2]

    @x.setter
    def x(self, value):
        """Set the x component of the vector."""
        self[0] = value

    @y.setter
    def y(self, value):
        """Set the y component of the vector."""
        self[1] = value

    @z.setter
    def z(self, value):
        """Set the z component of the vector."""
        self[2] = value

    def cross(self, other):
        """Return the cross product of this vector and another."""
        return Vector3(np.cross(self, other))

    def normalize(self):
        """Normalize the vector."""
        n = self / self.magnitude
        self.x = n.x
        self.y = n.y
        self.z = n.z

    def angle_to(self, other):
        """Return the angle between this vector and another."""
        return np.arccos(np.dot(self.normalized, other.normalized))

    def vector_to(self, other):
        """Return the vector from this vector to another."""
        return other - self

    def as_ndarray(self):
        """Return the vector as a numpy array."""
        return np.asarray(self)

    @property
    def normalized(self):
        """Return the normalized vector."""
        return self / self.magnitude

    @property
    def magnitude(self):
        """Return the magnitude of the vector."""
        return np.linalg.norm(self)
