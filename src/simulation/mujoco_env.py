#!/usr/bin/env python3
"""
MuJoCo Simulation Environment for Humanoid Robots
Physics simulation and visualization

Author: Al Numan
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
from pathlib import Path
import time

try:
    import mujoco
    import mujoco.viewer
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False
    print("Warning: MuJoCo not installed. Simulation will use simplified physics.")


@dataclass
class EnvConfig:
    """Simulation environment configuration"""
    model_path: Optional[str] = None
    timestep: float = 0.001           # Physics timestep (1 kHz)
    control_dt: float = 0.002         # Control rate (500 Hz)
    gravity: List[float] = field(default_factory=lambda: [0, 0, -9.81])
    ground_friction: float = 0.8
    render: bool = False
    render_fps: int = 60


@dataclass
class SimState:
    """Simulation state"""
    time: float
    qpos: np.ndarray               # Position (nq,)
    qvel: np.ndarray               # Velocity (nv,)
    ctrl: np.ndarray               # Control (nu,)
    contact_forces: Dict[str, np.ndarray] = field(default_factory=dict)


class MuJoCoEnv:
    """
    MuJoCo simulation environment for humanoid robots

    Provides:
    - Physics simulation
    - Contact force computation
    - Visualization
    - State reading/writing
    """

    def __init__(self, config: EnvConfig = None):
        """
        Initialize MuJoCo environment

        Args:
            config: Environment configuration
        """
        self.config = config or EnvConfig()

        if MUJOCO_AVAILABLE and self.config.model_path:
            self._init_mujoco()
        else:
            self._init_simple()

        # Simulation state
        self.sim_time = 0.0
        self.step_count = 0

        # Viewer
        self.viewer = None
        self.render_time = 0.0

    def _init_mujoco(self):
        """Initialize MuJoCo simulation"""
        self.model = mujoco.MjModel.from_xml_path(self.config.model_path)
        self.data = mujoco.MjData(self.model)

        # Set timestep
        self.model.opt.timestep = self.config.timestep

        # Set gravity
        self.model.opt.gravity[:] = self.config.gravity

        # Dimensions
        self.nq = self.model.nq
        self.nv = self.model.nv
        self.nu = self.model.nu

        # Contact body names (for force reading)
        self.contact_bodies = ['left_foot', 'right_foot']

        self.use_mujoco = True

    def _init_simple(self):
        """Initialize simple physics (no MuJoCo)"""
        self.model = None
        self.data = None

        # Default dimensions for H1
        self.nq = 26  # 7 (floating base) + 19 joints
        self.nv = 25  # 6 (floating base) + 19 joints
        self.nu = 19  # Actuated joints

        # State vectors
        self._qpos = np.zeros(self.nq)
        self._qpos[2] = 1.0  # Height
        self._qpos[6] = 1.0  # Quaternion w

        self._qvel = np.zeros(self.nv)
        self._ctrl = np.zeros(self.nu)

        self.use_mujoco = False

    def reset(self, qpos: np.ndarray = None, qvel: np.ndarray = None) -> SimState:
        """
        Reset simulation to initial state

        Args:
            qpos: Initial position (optional)
            qvel: Initial velocity (optional)

        Returns:
            Initial simulation state
        """
        self.sim_time = 0.0
        self.step_count = 0

        if self.use_mujoco:
            if qpos is not None:
                self.data.qpos[:] = qpos
            else:
                mujoco.mj_resetData(self.model, self.data)
                # Set to standing pose
                self.data.qpos[2] = 1.0  # Height
                self.data.qpos[6] = 1.0  # Quaternion w

            if qvel is not None:
                self.data.qvel[:] = qvel

            mujoco.mj_forward(self.model, self.data)

            return self._get_state()
        else:
            if qpos is not None:
                self._qpos = qpos.copy()
            else:
                self._qpos = np.zeros(self.nq)
                self._qpos[2] = 1.0
                self._qpos[6] = 1.0

            if qvel is not None:
                self._qvel = qvel.copy()
            else:
                self._qvel = np.zeros(self.nv)

            return self._get_state()

    def step(self, ctrl: np.ndarray) -> SimState:
        """
        Step simulation with control input

        Args:
            ctrl: Control input (joint torques or positions)

        Returns:
            New simulation state
        """
        if self.use_mujoco:
            # Apply control
            self.data.ctrl[:] = ctrl

            # Step physics
            n_steps = int(self.config.control_dt / self.config.timestep)
            for _ in range(n_steps):
                mujoco.mj_step(self.model, self.data)

            self.sim_time = self.data.time
        else:
            # Simple integration
            self._ctrl = ctrl.copy()
            self._simple_dynamics_step()
            self.sim_time += self.config.control_dt

        self.step_count += 1

        return self._get_state()

    def _simple_dynamics_step(self):
        """Simple dynamics integration (no MuJoCo)"""
        dt = self.config.control_dt

        # Extract state
        pos = self._qpos[:3]
        quat = self._qpos[3:7]
        joints = self._qpos[7:]

        vel = self._qvel[:3]
        omega = self._qvel[3:6]
        joint_vel = self._qvel[6:]

        # Gravity
        gravity = np.array(self.config.gravity)

        # Simple spring-damper ground contact
        if pos[2] < 1.0:
            # Ground reaction force
            k_ground = 10000
            b_ground = 1000
            f_ground = k_ground * (1.0 - pos[2]) - b_ground * vel[2]
            f_ground = max(f_ground, 0)

            accel = gravity + np.array([0, 0, f_ground / 47.0])
        else:
            accel = gravity

        # Integrate velocity
        vel = vel + accel * dt

        # Integrate position
        pos = pos + vel * dt

        # Keep above ground
        if pos[2] < 0.05:
            pos[2] = 0.05
            vel[2] = 0.0

        # Joint dynamics (simple PD + gravity compensation)
        joint_accel = (self._ctrl - 10.0 * joints - 2.0 * joint_vel) / 5.0
        joint_vel = joint_vel + joint_accel * dt
        joints = joints + joint_vel * dt

        # Update state
        self._qpos[:3] = pos
        self._qpos[7:] = joints
        self._qvel[:3] = vel
        self._qvel[6:] = joint_vel

    def _get_state(self) -> SimState:
        """Get current simulation state"""
        if self.use_mujoco:
            # Get contact forces
            contact_forces = self._get_contact_forces()

            return SimState(
                time=self.data.time,
                qpos=self.data.qpos.copy(),
                qvel=self.data.qvel.copy(),
                ctrl=self.data.ctrl.copy(),
                contact_forces=contact_forces
            )
        else:
            return SimState(
                time=self.sim_time,
                qpos=self._qpos.copy(),
                qvel=self._qvel.copy(),
                ctrl=self._ctrl.copy(),
                contact_forces={}
            )

    def _get_contact_forces(self) -> Dict[str, np.ndarray]:
        """Get contact forces for each foot"""
        forces = {}

        if not self.use_mujoco:
            return forces

        # Iterate through contacts
        for i in range(self.data.ncon):
            contact = self.data.contact[i]

            # Get body names
            geom1 = self.model.geom(contact.geom1).name
            geom2 = self.model.geom(contact.geom2).name

            # Check if this is a foot contact
            for foot in ['left_foot', 'right_foot']:
                if foot in geom1 or foot in geom2:
                    # Get contact force
                    force = np.zeros(6)
                    mujoco.mj_contactForce(self.model, self.data, i, force)
                    forces[foot] = force[:3]  # Linear force only

        return forces

    def get_body_state(self, body_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get pose and velocity of a body

        Args:
            body_name: Name of the body

        Returns:
            Tuple of (pose [7], velocity [6])
        """
        if self.use_mujoco:
            body_id = self.model.body(body_name).id
            pos = self.data.xpos[body_id]
            quat = self.data.xquat[body_id]
            pose = np.concatenate([pos, quat])

            vel = np.zeros(6)
            mujoco.mj_objectVelocity(
                self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
                body_id, vel, 0
            )

            return pose, vel
        else:
            # Return base state for simplified model
            pose = self._qpos[:7]
            vel = self._qvel[:6]
            return pose, vel

    def get_sensor_data(self) -> Dict[str, np.ndarray]:
        """Get sensor readings"""
        sensors = {}

        if self.use_mujoco:
            # IMU data
            if 'imu_acc' in [self.model.sensor(i).name for i in range(self.model.nsensor)]:
                acc_id = self.model.sensor('imu_acc').id
                sensors['imu_acceleration'] = self.data.sensordata[acc_id:acc_id+3].copy()

            if 'imu_gyro' in [self.model.sensor(i).name for i in range(self.model.nsensor)]:
                gyro_id = self.model.sensor('imu_gyro').id
                sensors['imu_angular_velocity'] = self.data.sensordata[gyro_id:gyro_id+3].copy()

            # Joint encoders
            sensors['joint_positions'] = self.data.qpos[7:].copy()
            sensors['joint_velocities'] = self.data.qvel[6:].copy()
        else:
            sensors['joint_positions'] = self._qpos[7:].copy()
            sensors['joint_velocities'] = self._qvel[6:].copy()

        return sensors

    def render_frame(self):
        """Render a single frame"""
        if not self.config.render:
            return

        if self.use_mujoco and self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        if self.viewer is not None:
            # Sync viewer
            self.viewer.sync()

            # Frame timing
            current_time = time.time()
            frame_time = 1.0 / self.config.render_fps
            if current_time - self.render_time < frame_time:
                time.sleep(frame_time - (current_time - self.render_time))
            self.render_time = time.time()

    def close(self):
        """Close simulation and viewer"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def set_external_force(
        self,
        body_name: str,
        force: np.ndarray,
        torque: np.ndarray = None
    ):
        """
        Apply external force to a body

        Args:
            body_name: Name of the body
            force: Force vector [fx, fy, fz]
            torque: Torque vector [tx, ty, tz]
        """
        if not self.use_mujoco:
            return

        body_id = self.model.body(body_name).id

        if torque is None:
            torque = np.zeros(3)

        # Apply force and torque
        self.data.xfrc_applied[body_id, :3] = force
        self.data.xfrc_applied[body_id, 3:] = torque

    def clear_external_forces(self):
        """Clear all external forces"""
        if self.use_mujoco:
            self.data.xfrc_applied[:] = 0

    @property
    def dt(self) -> float:
        """Get control timestep"""
        return self.config.control_dt

    @property
    def physics_dt(self) -> float:
        """Get physics timestep"""
        return self.config.timestep
