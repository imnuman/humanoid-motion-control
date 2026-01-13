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


# =============================================================
# Atlas 2026 Factory Functions
# =============================================================

def create_atlas_2026_model(
    config_path: Optional[str] = None,
    urdf_path: Optional[str] = None
) -> RobotModel:
    """
    Create Boston Dynamics Atlas 2026 robot model

    56 DOF fully rotational humanoid robot based on CES 2026 specs:
    - Height: 1.9m (6.2 ft)
    - Mass: 89 kg
    - Payload: 55 kg peak, 30 kg sustained
    - Battery: 4 hours runtime
    - IP67 rated

    Args:
        config_path: Path to Atlas config YAML (uses default if None)
        urdf_path: Path to Atlas URDF file (uses default if None)

    Returns:
        RobotModel configured for Atlas 2026
    """
    # Default paths
    if config_path is None:
        base_path = Path(__file__).parent.parent.parent
        config_path = str(base_path / "config" / "atlas_2026_config.yaml")

    if urdf_path is None:
        base_path = Path(__file__).parent.parent.parent
        urdf_path = str(base_path / "models" / "urdf" / "atlas_2026.urdf")

    # Create model
    model = RobotModel(
        urdf_path=urdf_path if Path(urdf_path).exists() else None,
        config_path=config_path if Path(config_path).exists() else None
    )

    # Override with Atlas-specific config if file doesn't exist
    if not Path(config_path).exists():
        model.config = RobotConfig(
            name="boston_dynamics_atlas_2026",
            total_mass=89.0,
            standing_height=1.9,
            num_joints=56,
            base_link="pelvis",
            left_foot="l_foot_link",
            right_foot="r_foot_link",
            left_hand="l_hand_link",
            right_hand="r_hand_link"
        )

    return model


def get_atlas_2026_mpc_params() -> Dict:
    """
    Get MPC parameters tuned for Atlas 2026 dynamics

    Returns:
        Dictionary of MPC parameters optimized for Atlas
    """
    return {
        # Robot physical parameters
        'mass': 89.0,
        'gravity': 9.81,
        'standing_height': 1.1,  # CoM height when standing
        'foot_size': (0.28, 0.12),  # Larger feet for stability
        'max_force': 800.0,  # Higher forces for heavier robot
        'friction_coefficient': 0.7,
        'hip_width': 0.32,
        'inertia': np.diag([12.0, 12.0, 3.0]),  # Higher inertia

        # MPC configuration
        'horizon': 15,  # Longer horizon for taller robot
        'dt': 0.02,  # 50 Hz

        # State tracking weights (tuned for stability)
        'Q_pos': np.array([20.0, 20.0, 200.0]),
        'Q_vel': np.array([5.0, 5.0, 20.0]),
        'Q_ori': np.array([100.0, 100.0, 20.0]),
        'Q_ang_vel': np.array([3.0, 3.0, 10.0]),

        # Control weights
        'R_force': 0.000005,
        'R_force_rate': 0.00005,
    }


def get_atlas_2026_wbc_params() -> Dict:
    """
    Get Whole-Body Control parameters for Atlas 2026

    Returns:
        Dictionary of WBC task weights and gains
    """
    return {
        'dt': 0.002,  # 500 Hz

        # Task weights (higher for taller, heavier robot)
        'com_tracking_weight': 150.0,
        'swing_foot_weight': 80.0,
        'torso_orientation_weight': 50.0,
        'arm_pose_weight': 20.0,
        'hand_pose_weight': 15.0,
        'head_tracking_weight': 10.0,
        'joint_regularization_weight': 0.5,

        # Gains for CoM tracking
        'com_kp': np.array([150.0, 150.0, 200.0]),
        'com_kd': np.array([30.0, 30.0, 40.0]),

        # Gains for swing foot
        'swing_kp': np.array([300.0, 300.0, 300.0, 150.0, 150.0, 150.0]),
        'swing_kd': np.array([60.0, 60.0, 60.0, 30.0, 30.0, 30.0]),

        # Gains for torso orientation
        'torso_kp': np.array([150.0, 150.0, 150.0]),
        'torso_kd': np.array([30.0, 30.0, 30.0]),

        # Gains for arms
        'arm_kp': 80.0,
        'arm_kd': 15.0,

        # Constraints
        'friction_cone': True,
        'torque_limits': True,
        'joint_limits': True,
        'self_collision': True,
    }


def get_atlas_2026_gait_params() -> Dict:
    """
    Get gait parameters optimized for Atlas 2026 (1.9m height)

    Returns:
        Dictionary of gait timing and step parameters
    """
    return {
        'walk': {
            'stance_duration': 0.45,  # Longer stance for stability
            'swing_duration': 0.35,
            'swing_height': 0.12,  # Higher step clearance
            'step_length': 0.50,  # Longer stride
            'phase_offset': 0.5,
        },
        'trot': {
            'stance_duration': 0.35,
            'swing_duration': 0.35,
            'swing_height': 0.15,
            'step_length': 0.60,
            'phase_offset': 0.5,
        },
        'run': {
            'stance_duration': 0.20,
            'swing_duration': 0.45,
            'swing_height': 0.20,
            'step_length': 0.80,
            'phase_offset': 0.5,
        },
        'industrial_walk': {
            # Slow, stable gait for payload handling
            'stance_duration': 0.60,
            'swing_duration': 0.40,
            'swing_height': 0.08,
            'step_length': 0.35,
            'phase_offset': 0.5,
        },
    }


def get_atlas_2026_joint_groups() -> Dict[str, List[str]]:
    """
    Get Atlas 2026 joint groupings for control

    Returns:
        Dictionary mapping body part names to joint lists
    """
    return {
        'left_leg': [
            'l_hip_yaw', 'l_hip_roll', 'l_hip_pitch',
            'l_knee_pitch', 'l_knee_roll',
            'l_ankle_pitch', 'l_ankle_roll', 'l_ankle_yaw'
        ],
        'right_leg': [
            'r_hip_yaw', 'r_hip_roll', 'r_hip_pitch',
            'r_knee_pitch', 'r_knee_roll',
            'r_ankle_pitch', 'r_ankle_roll', 'r_ankle_yaw'
        ],
        'torso': [
            'waist_yaw', 'waist_pitch', 'waist_roll',
            'chest_yaw', 'chest_pitch', 'chest_roll'
        ],
        'left_arm': [
            'l_shoulder_pitch', 'l_shoulder_roll', 'l_shoulder_yaw',
            'l_elbow_pitch', 'l_elbow_roll',
            'l_wrist_yaw', 'l_wrist_pitch', 'l_wrist_roll'
        ],
        'right_arm': [
            'r_shoulder_pitch', 'r_shoulder_roll', 'r_shoulder_yaw',
            'r_elbow_pitch', 'r_elbow_roll',
            'r_wrist_yaw', 'r_wrist_pitch', 'r_wrist_roll'
        ],
        'head': ['neck_yaw', 'neck_pitch', 'neck_roll'],
        'left_hand': [
            'l_thumb_base', 'l_thumb_flex',
            'l_index_flex', 'l_middle_flex', 'l_ring_flex', 'l_pinky_flex',
            'l_hand_spread'
        ],
        'right_hand': [
            'r_thumb_base', 'r_thumb_flex', 'r_thumb_tip',
            'r_index_flex', 'r_middle_flex', 'r_ring_flex', 'r_pinky_flex',
            'r_hand_spread'
        ],
        # Locomotion joints (used for walking)
        'locomotion': [
            'l_hip_yaw', 'l_hip_roll', 'l_hip_pitch',
            'l_knee_pitch', 'l_knee_roll',
            'l_ankle_pitch', 'l_ankle_roll', 'l_ankle_yaw',
            'r_hip_yaw', 'r_hip_roll', 'r_hip_pitch',
            'r_knee_pitch', 'r_knee_roll',
            'r_ankle_pitch', 'r_ankle_roll', 'r_ankle_yaw'
        ],
        # Upper body joints
        'upper_body': [
            'waist_yaw', 'waist_pitch', 'waist_roll',
            'chest_yaw', 'chest_pitch', 'chest_roll',
            'l_shoulder_pitch', 'l_shoulder_roll', 'l_shoulder_yaw',
            'l_elbow_pitch', 'l_elbow_roll',
            'l_wrist_yaw', 'l_wrist_pitch', 'l_wrist_roll',
            'r_shoulder_pitch', 'r_shoulder_roll', 'r_shoulder_yaw',
            'r_elbow_pitch', 'r_elbow_roll',
            'r_wrist_yaw', 'r_wrist_pitch', 'r_wrist_roll',
            'neck_yaw', 'neck_pitch', 'neck_roll'
        ],
    }


def get_atlas_2026_default_pose() -> Dict[str, float]:
    """
    Get default standing pose for Atlas 2026

    Returns:
        Dictionary of joint positions (radians)
    """
    return {
        # Left leg
        'l_hip_yaw': 0.0,
        'l_hip_roll': 0.0,
        'l_hip_pitch': -0.40,
        'l_knee_pitch': 0.80,
        'l_knee_roll': 0.0,
        'l_ankle_pitch': -0.40,
        'l_ankle_roll': 0.0,
        'l_ankle_yaw': 0.0,

        # Right leg
        'r_hip_yaw': 0.0,
        'r_hip_roll': 0.0,
        'r_hip_pitch': -0.40,
        'r_knee_pitch': 0.80,
        'r_knee_roll': 0.0,
        'r_ankle_pitch': -0.40,
        'r_ankle_roll': 0.0,
        'r_ankle_yaw': 0.0,

        # Torso
        'waist_yaw': 0.0,
        'waist_pitch': 0.0,
        'waist_roll': 0.0,
        'chest_yaw': 0.0,
        'chest_pitch': 0.0,
        'chest_roll': 0.0,

        # Left arm - Industrial ready pose
        'l_shoulder_pitch': 0.3,
        'l_shoulder_roll': 0.5,
        'l_shoulder_yaw': 0.0,
        'l_elbow_pitch': -1.2,
        'l_elbow_roll': 0.0,
        'l_wrist_yaw': 0.0,
        'l_wrist_pitch': 0.0,
        'l_wrist_roll': 0.0,

        # Right arm
        'r_shoulder_pitch': 0.3,
        'r_shoulder_roll': -0.5,
        'r_shoulder_yaw': 0.0,
        'r_elbow_pitch': -1.2,
        'r_elbow_roll': 0.0,
        'r_wrist_yaw': 0.0,
        'r_wrist_pitch': 0.0,
        'r_wrist_roll': 0.0,

        # Head
        'neck_yaw': 0.0,
        'neck_pitch': 0.0,
        'neck_roll': 0.0,

        # Left hand - Open
        'l_thumb_base': 0.3,
        'l_thumb_flex': 0.0,
        'l_index_flex': 0.0,
        'l_middle_flex': 0.0,
        'l_ring_flex': 0.0,
        'l_pinky_flex': 0.0,
        'l_hand_spread': 0.2,

        # Right hand - Open
        'r_thumb_base': 0.3,
        'r_thumb_flex': 0.0,
        'r_thumb_tip': 0.0,
        'r_index_flex': 0.0,
        'r_middle_flex': 0.0,
        'r_ring_flex': 0.0,
        'r_pinky_flex': 0.0,
        'r_hand_spread': 0.2,
    }
