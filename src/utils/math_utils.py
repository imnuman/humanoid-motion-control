#!/usr/bin/env python3
"""
Mathematical utilities for robotics
Rotation representations, transformations, and interpolation
"""

import numpy as np
from typing import Tuple, Union


def rotation_matrix_x(angle: float) -> np.ndarray:
    """Rotation matrix about X axis"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])


def rotation_matrix_y(angle: float) -> np.ndarray:
    """Rotation matrix about Y axis"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])


def rotation_matrix_z(angle: float) -> np.ndarray:
    """Rotation matrix about Z axis"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])


def euler_to_rotation_matrix(
    roll: float,
    pitch: float,
    yaw: float,
    order: str = 'xyz'
) -> np.ndarray:
    """
    Convert Euler angles to rotation matrix

    Args:
        roll: Rotation about X axis (radians)
        pitch: Rotation about Y axis (radians)
        yaw: Rotation about Z axis (radians)
        order: Rotation order (default 'xyz' for intrinsic rotations)

    Returns:
        3x3 rotation matrix
    """
    Rx = rotation_matrix_x(roll)
    Ry = rotation_matrix_y(pitch)
    Rz = rotation_matrix_z(yaw)

    if order == 'xyz':
        return Rz @ Ry @ Rx
    elif order == 'zyx':
        return Rx @ Ry @ Rz
    else:
        raise ValueError(f"Unsupported rotation order: {order}")


def rotation_matrix_to_euler(R: np.ndarray) -> Tuple[float, float, float]:
    """
    Extract Euler angles from rotation matrix (XYZ order)

    Args:
        R: 3x3 rotation matrix

    Returns:
        Tuple of (roll, pitch, yaw) in radians
    """
    # Handle gimbal lock
    if abs(R[2, 0]) >= 1.0 - 1e-6:
        yaw = 0.0
        if R[2, 0] < 0:
            pitch = np.pi / 2
            roll = np.arctan2(R[0, 1], R[0, 2])
        else:
            pitch = -np.pi / 2
            roll = np.arctan2(-R[0, 1], -R[0, 2])
    else:
        pitch = -np.arcsin(R[2, 0])
        roll = np.arctan2(R[2, 1] / np.cos(pitch), R[2, 2] / np.cos(pitch))
        yaw = np.arctan2(R[1, 0] / np.cos(pitch), R[0, 0] / np.cos(pitch))

    return roll, pitch, yaw


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to rotation matrix

    Args:
        q: Quaternion [w, x, y, z] (scalar first)

    Returns:
        3x3 rotation matrix
    """
    w, x, y, z = q

    # Normalize quaternion
    norm = np.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/norm, x/norm, y/norm, z/norm

    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])

    return R


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to quaternion

    Args:
        R: 3x3 rotation matrix

    Returns:
        Quaternion [w, x, y, z] (scalar first)
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    return np.array([w, x, y, z])


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """
    Create skew-symmetric matrix from vector

    Args:
        v: 3D vector

    Returns:
        3x3 skew-symmetric matrix such that skew(v) @ u = v x u
    """
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]"""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


def interpolate_pose(
    pose_start: np.ndarray,
    pose_end: np.ndarray,
    t: float
) -> np.ndarray:
    """
    Linearly interpolate between two poses

    Args:
        pose_start: Start pose [x, y, z, roll, pitch, yaw]
        pose_end: End pose
        t: Interpolation parameter [0, 1]

    Returns:
        Interpolated pose
    """
    t = np.clip(t, 0.0, 1.0)

    # Linear interpolation for position
    pos = pose_start[:3] + t * (pose_end[:3] - pose_start[:3])

    # Angle interpolation (handle wrapping)
    ori = np.zeros(3)
    for i in range(3):
        diff = normalize_angle(pose_end[3+i] - pose_start[3+i])
        ori[i] = normalize_angle(pose_start[3+i] + t * diff)

    return np.concatenate([pos, ori])


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """
    Spherical linear interpolation between quaternions

    Args:
        q0: Start quaternion [w, x, y, z]
        q1: End quaternion
        t: Interpolation parameter [0, 1]

    Returns:
        Interpolated quaternion
    """
    t = np.clip(t, 0.0, 1.0)

    # Normalize inputs
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)

    # Compute dot product
    dot = np.dot(q0, q1)

    # If negative dot, negate one quaternion for shortest path
    if dot < 0:
        q1 = -q1
        dot = -dot

    # If very close, use linear interpolation
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)

    # SLERP
    theta_0 = np.arccos(dot)
    theta = theta_0 * t

    q2 = q1 - q0 * dot
    q2 = q2 / np.linalg.norm(q2)

    return q0 * np.cos(theta) + q2 * np.sin(theta)


def compute_jacobian_numerical(
    func,
    x: np.ndarray,
    eps: float = 1e-6
) -> np.ndarray:
    """
    Compute Jacobian numerically using finite differences

    Args:
        func: Function mapping x -> y
        x: Input point
        eps: Perturbation size

    Returns:
        Jacobian matrix df/dx
    """
    f0 = func(x)
    n = len(x)
    m = len(f0)

    J = np.zeros((m, n))

    for i in range(n):
        x_plus = x.copy()
        x_plus[i] += eps
        f_plus = func(x_plus)
        J[:, i] = (f_plus - f0) / eps

    return J


def pose_to_transform(pose: np.ndarray) -> np.ndarray:
    """
    Convert pose [x, y, z, roll, pitch, yaw] to 4x4 homogeneous transform

    Args:
        pose: 6D pose vector

    Returns:
        4x4 transformation matrix
    """
    T = np.eye(4)
    T[:3, :3] = euler_to_rotation_matrix(pose[3], pose[4], pose[5])
    T[:3, 3] = pose[:3]
    return T


def transform_to_pose(T: np.ndarray) -> np.ndarray:
    """
    Convert 4x4 homogeneous transform to pose [x, y, z, roll, pitch, yaw]

    Args:
        T: 4x4 transformation matrix

    Returns:
        6D pose vector
    """
    pos = T[:3, 3]
    roll, pitch, yaw = rotation_matrix_to_euler(T[:3, :3])
    return np.array([pos[0], pos[1], pos[2], roll, pitch, yaw])


def adjoint_transform(T: np.ndarray) -> np.ndarray:
    """
    Compute 6x6 adjoint representation of SE(3) transform

    Args:
        T: 4x4 homogeneous transformation matrix

    Returns:
        6x6 adjoint matrix
    """
    R = T[:3, :3]
    p = T[:3, 3]

    Ad = np.zeros((6, 6))
    Ad[:3, :3] = R
    Ad[3:, 3:] = R
    Ad[3:, :3] = skew_symmetric(p) @ R

    return Ad
