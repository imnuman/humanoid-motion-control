"""Utility modules for humanoid control"""

from .math_utils import (
    rotation_matrix_x,
    rotation_matrix_y,
    rotation_matrix_z,
    euler_to_rotation_matrix,
    rotation_matrix_to_euler,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
    skew_symmetric,
    normalize_angle,
    interpolate_pose
)

from .robot_model import RobotModel

__all__ = [
    'rotation_matrix_x', 'rotation_matrix_y', 'rotation_matrix_z',
    'euler_to_rotation_matrix', 'rotation_matrix_to_euler',
    'quaternion_to_rotation_matrix', 'rotation_matrix_to_quaternion',
    'skew_symmetric', 'normalize_angle', 'interpolate_pose',
    'RobotModel'
]
