#!/usr/bin/env python3
"""
Task definitions for Whole-Body Control
Each task defines a desired behavior in task space
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

from ..utils.math_utils import skew_symmetric, rotation_matrix_to_euler


class TaskType(Enum):
    """Task types for prioritization"""
    CONTACT_CONSTRAINT = 0   # Highest priority - maintain contact
    COM_TRACKING = 1         # CoM position tracking
    SWING_FOOT = 2           # Swing foot trajectory
    TORSO_ORIENTATION = 3    # Torso orientation
    ARM_MOTION = 4           # Arm movements
    JOINT_REGULARIZATION = 5 # Lowest priority


@dataclass
class TaskGains:
    """PD gains for task-space control"""
    kp: np.ndarray  # Proportional gain
    kd: np.ndarray  # Derivative gain

    @classmethod
    def default_position(cls) -> 'TaskGains':
        return cls(
            kp=np.array([100.0, 100.0, 100.0]),
            kd=np.array([20.0, 20.0, 20.0])
        )

    @classmethod
    def default_orientation(cls) -> 'TaskGains':
        return cls(
            kp=np.array([100.0, 100.0, 100.0]),
            kd=np.array([20.0, 20.0, 20.0])
        )


class Task(ABC):
    """
    Abstract base class for whole-body control tasks

    Each task computes:
    - Task Jacobian J
    - Task-space error e
    - Desired acceleration (feedforward + feedback)
    """

    def __init__(
        self,
        name: str,
        dim: int,
        weight: float = 1.0,
        task_type: TaskType = TaskType.JOINT_REGULARIZATION
    ):
        """
        Initialize task

        Args:
            name: Task name for identification
            dim: Task dimension
            weight: Task weight for QP cost
            task_type: Task type for prioritization
        """
        self.name = name
        self.dim = dim
        self.weight = weight
        self.task_type = task_type
        self.active = True

    @abstractmethod
    def compute_error(
        self,
        q: np.ndarray,
        v: np.ndarray,
        robot_model
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute task error and error derivative

        Args:
            q: Joint positions
            v: Joint velocities
            robot_model: Robot model for kinematics

        Returns:
            Tuple of (position_error, velocity_error)
        """
        pass

    @abstractmethod
    def compute_jacobian(
        self,
        q: np.ndarray,
        robot_model
    ) -> np.ndarray:
        """
        Compute task Jacobian

        Args:
            q: Joint positions
            robot_model: Robot model

        Returns:
            Task Jacobian matrix (dim x nv)
        """
        pass

    def compute_desired_acceleration(
        self,
        q: np.ndarray,
        v: np.ndarray,
        robot_model,
        gains: TaskGains
    ) -> np.ndarray:
        """
        Compute desired task-space acceleration using PD control

        Args:
            q: Joint positions
            v: Joint velocities
            robot_model: Robot model
            gains: PD gains

        Returns:
            Desired acceleration in task space
        """
        pos_error, vel_error = self.compute_error(q, v, robot_model)

        # PD control law: a_des = kp * e + kd * e_dot + a_ff
        a_des = gains.kp * pos_error + gains.kd * vel_error

        return a_des

    def set_target(self, target: np.ndarray):
        """Set task target (to be overridden)"""
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}('{self.name}', dim={self.dim}, weight={self.weight})"


class CoMTask(Task):
    """Center of Mass tracking task"""

    def __init__(
        self,
        name: str = "com_tracking",
        weight: float = 100.0,
        target_position: np.ndarray = None,
        target_velocity: np.ndarray = None,
        gains: TaskGains = None
    ):
        super().__init__(name, dim=3, weight=weight, task_type=TaskType.COM_TRACKING)

        self.target_position = target_position if target_position is not None else np.zeros(3)
        self.target_velocity = target_velocity if target_velocity is not None else np.zeros(3)
        self.gains = gains or TaskGains.default_position()

    def set_target(
        self,
        position: np.ndarray = None,
        velocity: np.ndarray = None
    ):
        """Set desired CoM position and velocity"""
        if position is not None:
            self.target_position = position.copy()
        if velocity is not None:
            self.target_velocity = velocity.copy()

    def compute_error(
        self,
        q: np.ndarray,
        v: np.ndarray,
        robot_model
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute CoM position and velocity error"""
        current_pos = robot_model.get_com_position()
        current_vel = robot_model.get_com_velocity()

        pos_error = self.target_position - current_pos
        vel_error = self.target_velocity - current_vel

        return pos_error, vel_error

    def compute_jacobian(
        self,
        q: np.ndarray,
        robot_model
    ) -> np.ndarray:
        """Compute CoM Jacobian"""
        return robot_model.get_com_jacobian()


class FootPoseTask(Task):
    """Foot pose tracking task (position + orientation)"""

    def __init__(
        self,
        name: str,
        foot_frame: str,
        weight: float = 50.0,
        target_pose: np.ndarray = None,  # [x, y, z, roll, pitch, yaw]
        target_velocity: np.ndarray = None,  # [vx, vy, vz, wx, wy, wz]
        gains: TaskGains = None
    ):
        super().__init__(name, dim=6, weight=weight, task_type=TaskType.SWING_FOOT)

        self.foot_frame = foot_frame
        self.target_pose = target_pose if target_pose is not None else np.zeros(6)
        self.target_velocity = target_velocity if target_velocity is not None else np.zeros(6)
        self.gains = gains or TaskGains(
            kp=np.array([200.0, 200.0, 200.0, 100.0, 100.0, 100.0]),
            kd=np.array([40.0, 40.0, 40.0, 20.0, 20.0, 20.0])
        )

    def set_target(
        self,
        pose: np.ndarray = None,
        velocity: np.ndarray = None
    ):
        """Set desired foot pose and velocity"""
        if pose is not None:
            self.target_pose = pose.copy()
        if velocity is not None:
            self.target_velocity = velocity.copy()

    def compute_error(
        self,
        q: np.ndarray,
        v: np.ndarray,
        robot_model
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute foot pose error"""
        # Get current pose
        T = robot_model.get_frame_pose(self.foot_frame)
        current_pos = T[:3, 3]
        R = T[:3, :3]
        current_euler = np.array(rotation_matrix_to_euler(R))

        # Get current velocity
        current_vel = robot_model.get_frame_velocity(self.foot_frame)

        # Position error
        pos_error = self.target_pose[:3] - current_pos

        # Orientation error (simplified - direct Euler difference)
        ori_error = self.target_pose[3:6] - current_euler
        # Normalize angles
        ori_error = np.arctan2(np.sin(ori_error), np.cos(ori_error))

        # Full error
        error = np.concatenate([pos_error, ori_error])

        # Velocity error
        vel_error = self.target_velocity - current_vel

        return error, vel_error

    def compute_jacobian(
        self,
        q: np.ndarray,
        robot_model
    ) -> np.ndarray:
        """Compute foot Jacobian"""
        return robot_model.get_frame_jacobian(self.foot_frame)


class OrientationTask(Task):
    """Body orientation tracking task"""

    def __init__(
        self,
        name: str,
        frame: str,
        weight: float = 30.0,
        target_orientation: np.ndarray = None,  # [roll, pitch, yaw]
        target_angular_velocity: np.ndarray = None,
        gains: TaskGains = None
    ):
        super().__init__(name, dim=3, weight=weight, task_type=TaskType.TORSO_ORIENTATION)

        self.frame = frame
        self.target_orientation = (
            target_orientation if target_orientation is not None
            else np.zeros(3)
        )
        self.target_angular_velocity = (
            target_angular_velocity if target_angular_velocity is not None
            else np.zeros(3)
        )
        self.gains = gains or TaskGains.default_orientation()

    def set_target(
        self,
        orientation: np.ndarray = None,
        angular_velocity: np.ndarray = None
    ):
        """Set desired orientation"""
        if orientation is not None:
            self.target_orientation = orientation.copy()
        if angular_velocity is not None:
            self.target_angular_velocity = angular_velocity.copy()

    def compute_error(
        self,
        q: np.ndarray,
        v: np.ndarray,
        robot_model
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute orientation error"""
        T = robot_model.get_frame_pose(self.frame)
        R = T[:3, :3]
        current_euler = np.array(rotation_matrix_to_euler(R))

        # Orientation error
        error = self.target_orientation - current_euler
        # Normalize angles
        error = np.arctan2(np.sin(error), np.cos(error))

        # Angular velocity error
        current_vel = robot_model.get_frame_velocity(self.frame)
        vel_error = self.target_angular_velocity - current_vel[3:6]

        return error, vel_error

    def compute_jacobian(
        self,
        q: np.ndarray,
        robot_model
    ) -> np.ndarray:
        """Compute angular Jacobian (bottom 3 rows of frame Jacobian)"""
        J = robot_model.get_frame_jacobian(self.frame)
        return J[3:6, :]  # Angular part


class JointRegularizationTask(Task):
    """Joint position regularization task"""

    def __init__(
        self,
        name: str = "joint_regularization",
        num_joints: int = 19,
        weight: float = 1.0,
        target_position: np.ndarray = None,
        gains: TaskGains = None
    ):
        super().__init__(name, dim=num_joints, weight=weight,
                        task_type=TaskType.JOINT_REGULARIZATION)

        self.num_joints = num_joints
        self.target_position = (
            target_position if target_position is not None
            else np.zeros(num_joints)
        )
        self.gains = gains or TaskGains(
            kp=np.full(num_joints, 10.0),
            kd=np.full(num_joints, 2.0)
        )

    def set_target(self, position: np.ndarray):
        """Set target joint position"""
        self.target_position = position.copy()

    def compute_error(
        self,
        q: np.ndarray,
        v: np.ndarray,
        robot_model
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute joint position error"""
        # Extract joint positions (skip floating base)
        current_pos = q[7:7+self.num_joints] if len(q) > self.num_joints else q
        current_vel = v[6:6+self.num_joints] if len(v) > self.num_joints else v

        pos_error = self.target_position - current_pos
        vel_error = -current_vel  # Target velocity is zero

        return pos_error, vel_error

    def compute_jacobian(
        self,
        q: np.ndarray,
        robot_model
    ) -> np.ndarray:
        """Joint regularization Jacobian is identity for joints"""
        nv = robot_model.nv if hasattr(robot_model, 'nv') else 6 + self.num_joints

        J = np.zeros((self.num_joints, nv))
        # Identity for joint positions
        J[:, 6:6+self.num_joints] = np.eye(self.num_joints)

        return J


class ContactTask(Task):
    """Contact constraint task - maintains zero velocity at contact"""

    def __init__(
        self,
        name: str,
        contact_frame: str,
        weight: float = 1000.0,  # High weight for constraint
    ):
        super().__init__(name, dim=6, weight=weight,
                        task_type=TaskType.CONTACT_CONSTRAINT)

        self.contact_frame = contact_frame

    def compute_error(
        self,
        q: np.ndarray,
        v: np.ndarray,
        robot_model
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Contact should have zero velocity"""
        current_vel = robot_model.get_frame_velocity(self.contact_frame)
        return np.zeros(6), -current_vel

    def compute_jacobian(
        self,
        q: np.ndarray,
        robot_model
    ) -> np.ndarray:
        """Contact Jacobian"""
        return robot_model.get_frame_jacobian(self.contact_frame)
