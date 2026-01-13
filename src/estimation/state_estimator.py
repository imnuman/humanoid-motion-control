#!/usr/bin/env python3
"""
EKF State Estimator for Humanoid Robots
Fuses IMU and leg kinematics for base state estimation

Author: Al Numan
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple

from ..utils.math_utils import (
    euler_to_rotation_matrix,
    rotation_matrix_to_euler,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
    skew_symmetric
)


@dataclass
class IMUData:
    """IMU sensor data"""
    acceleration: np.ndarray      # Linear acceleration (body frame)
    angular_velocity: np.ndarray  # Angular velocity (body frame)
    orientation: Optional[np.ndarray] = None  # Optional orientation estimate (quaternion)
    timestamp: float = 0.0


@dataclass
class JointState:
    """Joint encoder data"""
    positions: np.ndarray
    velocities: np.ndarray
    timestamp: float = 0.0


@dataclass
class RobotState:
    """Complete robot state estimate"""
    # Base pose and velocity
    position: np.ndarray           # World frame position
    velocity: np.ndarray           # World frame velocity
    orientation: np.ndarray        # Quaternion [w, x, y, z]
    angular_velocity: np.ndarray   # Body frame angular velocity

    # Joint state
    joint_positions: np.ndarray
    joint_velocities: np.ndarray

    # Covariance
    covariance: Optional[np.ndarray] = None

    # Contact state
    contact_state: Dict[str, bool] = field(default_factory=dict)

    @classmethod
    def zero(cls, num_joints: int = 19) -> 'RobotState':
        """Create zero state"""
        return cls(
            position=np.array([0.0, 0.0, 1.0]),
            velocity=np.zeros(3),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_velocity=np.zeros(3),
            joint_positions=np.zeros(num_joints),
            joint_velocities=np.zeros(num_joints)
        )

    def get_euler_orientation(self) -> np.ndarray:
        """Get orientation as Euler angles [roll, pitch, yaw]"""
        R = quaternion_to_rotation_matrix(self.orientation)
        return np.array(rotation_matrix_to_euler(R))

    def get_rotation_matrix(self) -> np.ndarray:
        """Get rotation matrix from world to body"""
        return quaternion_to_rotation_matrix(self.orientation)


@dataclass
class EstimatorConfig:
    """State estimator configuration"""
    # Process noise
    process_noise_position: float = 0.001
    process_noise_velocity: float = 0.01
    process_noise_orientation: float = 0.001
    process_noise_gyro_bias: float = 0.0001
    process_noise_accel_bias: float = 0.0001

    # Measurement noise
    measurement_noise_kinematics: float = 0.01
    measurement_noise_velocity: float = 0.05

    # IMU parameters
    gravity: float = 9.81
    imu_frame: str = "base_link"

    # Contact frames
    contact_frames: List[str] = field(default_factory=lambda: ['left_foot', 'right_foot'])


class StateEstimator:
    """
    Extended Kalman Filter for robot state estimation

    State vector: x = [p, v, q, b_g, b_a] ∈ R^16
    - p: position (3)
    - v: velocity (3)
    - q: orientation quaternion (4)
    - b_g: gyroscope bias (3)
    - b_a: accelerometer bias (3)

    Predictions:
    - IMU for velocity and orientation

    Updates:
    - Leg kinematics for position (contact feet are fixed points)
    """

    def __init__(
        self,
        robot_model,
        config: EstimatorConfig = None
    ):
        """
        Initialize state estimator

        Args:
            robot_model: Robot model for forward kinematics
            config: Estimator configuration
        """
        self.robot = robot_model
        self.config = config or EstimatorConfig()

        # State dimension
        self.state_dim = 16  # [p(3), v(3), q(4), b_g(3), b_a(3)]

        # State vector
        self.x = np.zeros(self.state_dim)
        self.x[2] = 1.0  # Initial height
        self.x[6] = 1.0  # Quaternion w = 1

        # Covariance matrix
        self.P = np.eye(self.state_dim) * 0.1

        # Process noise
        self._setup_process_noise()

        # Contact state
        self.contact_state = {frame: False for frame in self.config.contact_frames}
        self.contact_positions = {frame: np.zeros(3) for frame in self.config.contact_frames}

        # Previous update time
        self.last_update_time = 0.0

        # Gravity vector in world frame
        self.g_world = np.array([0, 0, -self.config.gravity])

    def _setup_process_noise(self):
        """Setup process noise covariance"""
        self.Q = np.diag([
            # Position
            self.config.process_noise_position,
            self.config.process_noise_position,
            self.config.process_noise_position,
            # Velocity
            self.config.process_noise_velocity,
            self.config.process_noise_velocity,
            self.config.process_noise_velocity,
            # Orientation (quaternion)
            self.config.process_noise_orientation,
            self.config.process_noise_orientation,
            self.config.process_noise_orientation,
            self.config.process_noise_orientation,
            # Gyro bias
            self.config.process_noise_gyro_bias,
            self.config.process_noise_gyro_bias,
            self.config.process_noise_gyro_bias,
            # Accel bias
            self.config.process_noise_accel_bias,
            self.config.process_noise_accel_bias,
            self.config.process_noise_accel_bias,
        ])

    def predict(self, imu: IMUData, dt: float):
        """
        Prediction step using IMU measurements

        Args:
            imu: IMU sensor data
            dt: Time step
        """
        # Extract state
        p = self.x[0:3]
        v = self.x[3:6]
        q = self.x[6:10]
        b_g = self.x[10:13]
        b_a = self.x[13:16]

        # Rotation matrix from body to world
        R = quaternion_to_rotation_matrix(q)

        # Corrected IMU measurements
        omega = imu.angular_velocity - b_g
        accel = imu.acceleration - b_a

        # Predict position
        p_new = p + v * dt + 0.5 * (R @ accel + self.g_world) * dt * dt

        # Predict velocity
        v_new = v + (R @ accel + self.g_world) * dt

        # Predict orientation (quaternion integration)
        omega_mag = np.linalg.norm(omega)
        if omega_mag > 1e-10:
            # Axis-angle to quaternion
            axis = omega / omega_mag
            angle = omega_mag * dt
            dq = np.array([
                np.cos(angle / 2),
                axis[0] * np.sin(angle / 2),
                axis[1] * np.sin(angle / 2),
                axis[2] * np.sin(angle / 2)
            ])
            # Quaternion multiplication
            q_new = self._quaternion_multiply(q, dq)
            q_new = q_new / np.linalg.norm(q_new)  # Normalize
        else:
            q_new = q.copy()

        # Biases remain constant (random walk)
        b_g_new = b_g
        b_a_new = b_a

        # Update state
        self.x[0:3] = p_new
        self.x[3:6] = v_new
        self.x[6:10] = q_new
        self.x[10:13] = b_g_new
        self.x[13:16] = b_a_new

        # Propagate covariance
        F = self._compute_jacobian_f(R, accel, omega, dt)
        self.P = F @ self.P @ F.T + self.Q * dt

    def update_kinematics(
        self,
        joint_state: JointState,
        contact_state: Dict[str, bool]
    ):
        """
        Update step using leg kinematics

        For feet in contact, the foot position in world frame is constant.
        This provides an observation of base position.

        Args:
            joint_state: Joint encoder data
            contact_state: Which feet are in contact
        """
        self.contact_state = contact_state.copy()

        # Update robot model
        q_robot = self._build_robot_config(joint_state)
        self.robot.update_state(q_robot, np.zeros(self.robot.nv))

        # Get current orientation
        R = quaternion_to_rotation_matrix(self.x[6:10])

        # Process each contact
        for frame, in_contact in contact_state.items():
            if not in_contact:
                continue

            # Get foot position relative to base
            T_foot = self.robot.get_frame_pose(frame)
            p_foot_body = T_foot[:3, 3]

            # Foot position in world = base position + R * foot_in_body
            # If foot is fixed, then: p_base = p_foot_world - R * foot_in_body
            # This is our measurement

            if self._is_new_contact(frame, contact_state):
                # Store contact position
                p_base = self.x[0:3]
                self.contact_positions[frame] = p_base + R @ p_foot_body
            else:
                # Use stored contact position
                p_foot_world = self.contact_positions[frame]

                # Expected base position from contact
                z = p_foot_world - R @ p_foot_body

                # Measurement model: z = h(x) + noise
                # h(x) = p_foot_world - R(q) * p_foot_body
                # For this update: y = z - p_base (innovation)

                y = z - self.x[0:3]

                # Measurement Jacobian
                H = np.zeros((3, self.state_dim))
                H[0:3, 0:3] = np.eye(3)  # Position
                # Also depends on orientation but we linearize

                # Measurement noise
                R_meas = np.eye(3) * self.config.measurement_noise_kinematics

                # Kalman gain
                S = H @ self.P @ H.T + R_meas
                K = self.P @ H.T @ np.linalg.inv(S)

                # State update
                self.x = self.x + K @ y

                # Covariance update
                self.P = (np.eye(self.state_dim) - K @ H) @ self.P

        # Normalize quaternion
        self.x[6:10] = self.x[6:10] / np.linalg.norm(self.x[6:10])

    def update(
        self,
        imu: IMUData,
        joint_state: JointState,
        contact_state: Dict[str, bool],
        dt: float
    ):
        """
        Full EKF update step

        Args:
            imu: IMU sensor data
            joint_state: Joint encoder data
            contact_state: Contact state for each foot
            dt: Time step
        """
        # Prediction with IMU
        self.predict(imu, dt)

        # Update with kinematics
        self.update_kinematics(joint_state, contact_state)

    def get_state(self) -> RobotState:
        """Get current state estimate"""
        return RobotState(
            position=self.x[0:3].copy(),
            velocity=self.x[3:6].copy(),
            orientation=self.x[6:10].copy(),
            angular_velocity=self.x[10:13].copy(),  # Use gyro bias estimate
            joint_positions=np.zeros(19),  # Would be stored separately
            joint_velocities=np.zeros(19),
            covariance=self.P.copy(),
            contact_state=self.contact_state.copy()
        )

    def reset(
        self,
        position: np.ndarray = None,
        orientation: np.ndarray = None
    ):
        """Reset estimator state"""
        self.x = np.zeros(self.state_dim)
        self.x[2] = 1.0 if position is None else position[2]
        self.x[6] = 1.0  # Quaternion w

        if position is not None:
            self.x[0:3] = position

        if orientation is not None:
            self.x[6:10] = orientation

        self.P = np.eye(self.state_dim) * 0.1
        self.contact_positions = {frame: np.zeros(3) for frame in self.config.contact_frames}

    def _build_robot_config(self, joint_state: JointState) -> np.ndarray:
        """Build full robot configuration from joint state"""
        # Floating base position and orientation
        q = np.zeros(self.robot.nq)
        q[0:3] = self.x[0:3]  # Position
        q[3:7] = self.x[6:10]  # Quaternion

        # Joint positions
        n_joints = len(joint_state.positions)
        q[7:7+n_joints] = joint_state.positions

        return q

    def _is_new_contact(
        self,
        frame: str,
        contact_state: Dict[str, bool]
    ) -> bool:
        """Check if this is a new contact event"""
        # Simple check - would track previous state in practice
        return np.linalg.norm(self.contact_positions[frame]) < 1e-6

    def _compute_jacobian_f(
        self,
        R: np.ndarray,
        accel: np.ndarray,
        omega: np.ndarray,
        dt: float
    ) -> np.ndarray:
        """Compute state transition Jacobian"""
        F = np.eye(self.state_dim)

        # dp/dv
        F[0:3, 3:6] = np.eye(3) * dt

        # dv/dq (linearized)
        # dv/db_a
        F[3:6, 13:16] = -R * dt

        return F

    def _quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions [w, x, y, z]"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
