#!/usr/bin/env python3
"""
Robot Model Wrapper
Pinocchio-based rigid body dynamics computation
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
import yaml
from pathlib import Path

try:
    import pinocchio as pin
    PINOCCHIO_AVAILABLE = True
except ImportError:
    PINOCCHIO_AVAILABLE = False
    print("Warning: Pinocchio not installed. Using simplified dynamics.")


@dataclass
class JointLimits:
    """Joint position, velocity, and torque limits"""
    position_lower: np.ndarray
    position_upper: np.ndarray
    velocity: np.ndarray
    torque: np.ndarray


@dataclass
class RobotConfig:
    """Robot configuration parameters"""
    name: str = "humanoid"
    total_mass: float = 47.0
    standing_height: float = 1.0
    num_joints: int = 19

    # Frame names
    base_link: str = "base_link"
    left_foot: str = "left_foot"
    right_foot: str = "right_foot"
    left_hand: str = "left_hand"
    right_hand: str = "right_hand"

    # Joint groups
    left_leg_joints: List[str] = field(default_factory=list)
    right_leg_joints: List[str] = field(default_factory=list)
    left_arm_joints: List[str] = field(default_factory=list)
    right_arm_joints: List[str] = field(default_factory=list)
    torso_joints: List[str] = field(default_factory=list)


class RobotModel:
    """
    Rigid body dynamics model using Pinocchio

    Provides:
    - Forward kinematics
    - Inverse kinematics
    - Mass matrix computation
    - Coriolis and gravity terms
    - Jacobian computation
    - Contact dynamics
    """

    def __init__(
        self,
        urdf_path: Optional[str] = None,
        config_path: Optional[str] = None
    ):
        """
        Initialize robot model

        Args:
            urdf_path: Path to URDF file
            config_path: Path to YAML configuration file
        """
        self.config = RobotConfig()

        if config_path:
            self._load_config(config_path)

        # Initialize Pinocchio model if available
        if PINOCCHIO_AVAILABLE and urdf_path:
            self.model = pin.buildModelFromUrdf(urdf_path)
            self.data = self.model.createData()
            self.nq = self.model.nq  # Position dimension
            self.nv = self.model.nv  # Velocity dimension
        else:
            # Simplified model for testing
            self.model = None
            self.data = None
            self.nq = self.config.num_joints + 7  # joints + floating base
            self.nv = self.config.num_joints + 6

        # Allocate state vectors
        self.q = np.zeros(self.nq)
        self.v = np.zeros(self.nv)
        self.a = np.zeros(self.nv)

        # Set default configuration
        self._set_default_configuration()

        # Contact frames
        self.contact_frames = {
            'left_foot': self.config.left_foot,
            'right_foot': self.config.right_foot
        }

    def _load_config(self, config_path: str):
        """Load robot configuration from YAML"""
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)

        robot_cfg = cfg.get('robot', {})
        self.config.name = robot_cfg.get('name', 'humanoid')
        self.config.total_mass = robot_cfg.get('total_mass', 47.0)
        self.config.standing_height = robot_cfg.get('standing_height', 1.0)
        self.config.num_joints = robot_cfg.get('num_joints', 19)

        frames = cfg.get('frames', {})
        self.config.base_link = frames.get('base', 'base_link')
        self.config.left_foot = frames.get('left_foot', 'left_foot')
        self.config.right_foot = frames.get('right_foot', 'right_foot')

    def _set_default_configuration(self):
        """Set default standing configuration"""
        if self.model is not None and PINOCCHIO_AVAILABLE:
            # Use Pinocchio neutral configuration
            self.q = pin.neutral(self.model)
        else:
            # Set floating base to standing height
            self.q[2] = self.config.standing_height
            self.q[6] = 1.0  # Quaternion w = 1

    def update_state(
        self,
        q: np.ndarray,
        v: np.ndarray,
        a: Optional[np.ndarray] = None
    ):
        """Update robot state"""
        self.q = q.copy()
        self.v = v.copy()
        if a is not None:
            self.a = a.copy()

        if self.model is not None and PINOCCHIO_AVAILABLE:
            pin.forwardKinematics(self.model, self.data, self.q, self.v, self.a)
            pin.updateFramePlacements(self.model, self.data)

    def get_frame_pose(self, frame_name: str) -> np.ndarray:
        """
        Get frame pose in world coordinates

        Args:
            frame_name: Name of the frame

        Returns:
            4x4 homogeneous transformation matrix
        """
        if self.model is not None and PINOCCHIO_AVAILABLE:
            frame_id = self.model.getFrameId(frame_name)
            oMf = self.data.oMf[frame_id]
            T = np.eye(4)
            T[:3, :3] = oMf.rotation
            T[:3, 3] = oMf.translation
            return T
        else:
            # Simplified: return identity for testing
            T = np.eye(4)
            T[2, 3] = self.config.standing_height
            return T

    def get_frame_velocity(self, frame_name: str) -> np.ndarray:
        """
        Get frame velocity (linear and angular)

        Args:
            frame_name: Name of the frame

        Returns:
            6D velocity vector [linear, angular]
        """
        if self.model is not None and PINOCCHIO_AVAILABLE:
            frame_id = self.model.getFrameId(frame_name)
            v = pin.getFrameVelocity(
                self.model, self.data, frame_id,
                pin.ReferenceFrame.WORLD
            )
            return np.concatenate([v.linear, v.angular])
        else:
            return np.zeros(6)

    def get_frame_jacobian(
        self,
        frame_name: str,
        reference_frame: str = 'world'
    ) -> np.ndarray:
        """
        Compute frame Jacobian

        Args:
            frame_name: Name of the frame
            reference_frame: 'world' or 'local'

        Returns:
            6 x nv Jacobian matrix
        """
        if self.model is not None and PINOCCHIO_AVAILABLE:
            frame_id = self.model.getFrameId(frame_name)
            if reference_frame == 'world':
                rf = pin.ReferenceFrame.WORLD
            else:
                rf = pin.ReferenceFrame.LOCAL

            J = pin.computeFrameJacobian(
                self.model, self.data, self.q, frame_id, rf
            )
            return J
        else:
            # Simplified Jacobian for testing
            return np.eye(6, self.nv)

    def get_mass_matrix(self) -> np.ndarray:
        """
        Compute joint-space mass matrix M(q)

        Returns:
            nv x nv mass matrix
        """
        if self.model is not None and PINOCCHIO_AVAILABLE:
            M = pin.crba(self.model, self.data, self.q)
            return M
        else:
            # Simplified diagonal mass matrix
            return np.eye(self.nv) * 5.0

    def get_nonlinear_effects(self) -> np.ndarray:
        """
        Compute Coriolis, centrifugal, and gravity terms h(q, v)

        Returns:
            nv-dimensional vector
        """
        if self.model is not None and PINOCCHIO_AVAILABLE:
            h = pin.nonLinearEffects(self.model, self.data, self.q, self.v)
            return h
        else:
            # Simplified gravity compensation
            g = np.zeros(self.nv)
            g[2] = -self.config.total_mass * 9.81
            return g

    def get_gravity_vector(self) -> np.ndarray:
        """
        Compute gravity compensation torques g(q)

        Returns:
            nv-dimensional gravity vector
        """
        if self.model is not None and PINOCCHIO_AVAILABLE:
            g = pin.computeGeneralizedGravity(self.model, self.data, self.q)
            return g
        else:
            g = np.zeros(self.nv)
            g[2] = -self.config.total_mass * 9.81
            return g

    def get_centroidal_momentum(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute centroidal momentum (linear and angular)

        Returns:
            Tuple of (linear_momentum, angular_momentum)
        """
        if self.model is not None and PINOCCHIO_AVAILABLE:
            h = pin.computeCentroidalMomentum(self.model, self.data, self.q, self.v)
            return h.linear, h.angular
        else:
            linear = self.config.total_mass * self.v[:3]
            angular = np.zeros(3)
            return linear, angular

    def get_com_position(self) -> np.ndarray:
        """Get center of mass position in world frame"""
        if self.model is not None and PINOCCHIO_AVAILABLE:
            com = pin.centerOfMass(self.model, self.data, self.q)
            return com
        else:
            return np.array([0.0, 0.0, self.config.standing_height])

    def get_com_velocity(self) -> np.ndarray:
        """Get center of mass velocity in world frame"""
        if self.model is not None and PINOCCHIO_AVAILABLE:
            pin.centerOfMass(self.model, self.data, self.q, self.v)
            return self.data.vcom[0]
        else:
            return self.v[:3]

    def get_com_jacobian(self) -> np.ndarray:
        """Compute CoM Jacobian"""
        if self.model is not None and PINOCCHIO_AVAILABLE:
            Jcom = pin.jacobianCenterOfMass(self.model, self.data, self.q)
            return Jcom
        else:
            # Simplified: CoM moves with base
            J = np.zeros((3, self.nv))
            J[:, :3] = np.eye(3)
            return J

    def inverse_dynamics(
        self,
        q: np.ndarray,
        v: np.ndarray,
        a: np.ndarray
    ) -> np.ndarray:
        """
        Compute inverse dynamics: tau = M(q) * a + h(q, v)

        Args:
            q: Joint positions
            v: Joint velocities
            a: Joint accelerations

        Returns:
            Joint torques
        """
        if self.model is not None and PINOCCHIO_AVAILABLE:
            tau = pin.rnea(self.model, self.data, q, v, a)
            return tau
        else:
            M = self.get_mass_matrix()
            h = self.get_nonlinear_effects()
            return M @ a + h

    def forward_dynamics(
        self,
        q: np.ndarray,
        v: np.ndarray,
        tau: np.ndarray
    ) -> np.ndarray:
        """
        Compute forward dynamics: a = M(q)^-1 * (tau - h(q, v))

        Args:
            q: Joint positions
            v: Joint velocities
            tau: Joint torques

        Returns:
            Joint accelerations
        """
        if self.model is not None and PINOCCHIO_AVAILABLE:
            a = pin.aba(self.model, self.data, q, v, tau)
            return a
        else:
            M = self.get_mass_matrix()
            h = self.get_nonlinear_effects()
            return np.linalg.solve(M, tau - h)

    def get_contact_jacobian(
        self,
        contact_frames: List[str]
    ) -> np.ndarray:
        """
        Stack Jacobians for multiple contact frames

        Args:
            contact_frames: List of frame names in contact

        Returns:
            Stacked Jacobian matrix
        """
        jacobians = []
        for frame in contact_frames:
            J = self.get_frame_jacobian(frame)
            jacobians.append(J)

        if jacobians:
            return np.vstack(jacobians)
        else:
            return np.zeros((0, self.nv))

    def get_foot_positions(self) -> Dict[str, np.ndarray]:
        """Get both foot positions in world frame"""
        return {
            'left': self.get_frame_pose(self.config.left_foot)[:3, 3],
            'right': self.get_frame_pose(self.config.right_foot)[:3, 3]
        }

    def get_joint_limits(self) -> JointLimits:
        """Get joint limits from model"""
        if self.model is not None and PINOCCHIO_AVAILABLE:
            return JointLimits(
                position_lower=self.model.lowerPositionLimit[7:],
                position_upper=self.model.upperPositionLimit[7:],
                velocity=self.model.velocityLimit[6:],
                torque=self.model.effortLimit[6:]
            )
        else:
            n = self.config.num_joints
            return JointLimits(
                position_lower=np.full(n, -np.pi),
                position_upper=np.full(n, np.pi),
                velocity=np.full(n, 10.0),
                torque=np.full(n, 100.0)
            )

    @property
    def total_mass(self) -> float:
        """Get total robot mass"""
        if self.model is not None and PINOCCHIO_AVAILABLE:
            mass = 0.0
            for inertia in self.model.inertias:
                mass += inertia.mass
            return mass
        else:
            return self.config.total_mass

    @property
    def joint_names(self) -> List[str]:
        """Get list of joint names"""
        if self.model is not None and PINOCCHIO_AVAILABLE:
            return [self.model.names[i] for i in range(1, self.model.njoints)]
        else:
            return [f"joint_{i}" for i in range(self.config.num_joints)]


def create_simple_humanoid_model(
    standing_height: float = 1.0,
    mass: float = 47.0
) -> RobotModel:
    """
    Create simplified humanoid model for testing without URDF

    Args:
        standing_height: Nominal standing height
        mass: Total robot mass

    Returns:
        RobotModel instance
    """
    config = RobotConfig(
        name="simple_humanoid",
        total_mass=mass,
        standing_height=standing_height,
        num_joints=19
    )

    model = RobotModel()
    model.config = config
    return model
