#!/usr/bin/env python3
"""
Isaac Gym Environment for Humanoid Locomotion
Vectorized simulation for RL training

Author: Al Numan
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional

import torch

# Isaac Gym import (conditional)
try:
    from isaacgym import gymapi, gymtorch, gymutil
    ISAACGYM_AVAILABLE = True
except ImportError:
    ISAACGYM_AVAILABLE = False
    print("Warning: Isaac Gym not installed. Using simplified environment.")


@dataclass
class EnvConfig:
    """Environment configuration"""
    num_envs: int = 4096
    env_spacing: float = 3.0

    # Simulation
    sim_dt: float = 0.002
    control_dt: float = 0.02
    gravity: List[float] = field(default_factory=lambda: [0, 0, -9.81])

    # Robot
    robot_asset: str = "h1.urdf"
    num_joints: int = 19

    # Observation
    obs_scales: Dict[str, float] = field(default_factory=lambda: {
        'base_pos': 1.0,
        'base_vel': 2.0,
        'base_ang_vel': 0.25,
        'joint_pos': 1.0,
        'joint_vel': 0.05,
        'command': 1.0
    })

    # Action
    action_scale: float = 0.5
    clip_actions: float = 1.0

    # Reward
    reward_scales: Dict[str, float] = field(default_factory=lambda: {
        'tracking_linear_vel': 1.0,
        'tracking_angular_vel': 0.5,
        'feet_air_time': 1.0,
        'base_height': -1.0,
        'torques': -0.00001,
        'dof_vel': -0.0001,
        'dof_acc': -2.5e-7,
        'action_rate': -0.01,
        'collision': -1.0,
        'termination': -0.0,
        'alive': 0.15
    })

    # Termination
    termination_height: float = 0.3
    termination_body_contact: List[str] = field(default_factory=lambda: ['torso', 'knee'])

    # Domain randomization
    randomize: bool = True
    randomize_friction: Tuple[float, float] = (0.5, 1.25)
    randomize_mass: Tuple[float, float] = (0.9, 1.1)
    randomize_motor_strength: Tuple[float, float] = (0.9, 1.1)


class HumanoidEnv:
    """
    Isaac Gym environment for humanoid locomotion

    Features:
    - Vectorized simulation for massively parallel training
    - Domain randomization
    - Customizable reward structure
    - Terrain generation
    """

    def __init__(self, config: EnvConfig = None, device: str = "cuda"):
        """
        Initialize environment

        Args:
            config: Environment configuration
            device: Compute device ("cuda" or "cpu")
        """
        self.config = config or EnvConfig()
        self.device = torch.device(device)

        self.num_envs = self.config.num_envs
        self.num_obs = self._compute_obs_dim()
        self.num_actions = self.config.num_joints

        # Episode tracking
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device)
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device)

        # Commands
        self.commands = torch.zeros(self.num_envs, 3, device=self.device)  # [vx, vy, yaw_rate]

        if ISAACGYM_AVAILABLE:
            self._init_isaac_gym()
        else:
            self._init_simple()

        # Default joint positions
        self.default_dof_pos = torch.zeros(self.num_actions, device=self.device)

        # Action buffer for smoothing
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, device=self.device)

    def _compute_obs_dim(self) -> int:
        """Compute observation dimension"""
        # Base: pos(3) + ori(3) + vel(3) + ang_vel(3) = 12
        # Joints: pos(n) + vel(n) = 2n
        # Commands: 3
        # Previous action: n
        n = self.config.num_joints
        return 12 + 2 * n + 3 + n

    def _init_isaac_gym(self):
        """Initialize Isaac Gym simulation"""
        # Create gym instance
        self.gym = gymapi.acquire_gym()

        # Parse arguments
        args = gymutil.parse_arguments(description="Humanoid Training")

        # Create sim
        sim_params = gymapi.SimParams()
        sim_params.dt = self.config.sim_dt
        sim_params.substeps = 2
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(*self.config.gravity)

        # PhysX settings
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.contact_offset = 0.01
        sim_params.physx.rest_offset = 0.0

        # Enable GPU
        sim_params.use_gpu_pipeline = (self.device.type == "cuda")

        self.sim = self.gym.create_sim(
            args.compute_device_id,
            args.graphics_device_id,
            gymapi.SIM_PHYSX,
            sim_params
        )

        # Create ground plane
        self._create_ground_plane()

        # Create environments
        self._create_envs()

        # Prepare tensors
        self.gym.prepare_sim(self.sim)
        self._create_tensors()

    def _init_simple(self):
        """Initialize simple simulation (no Isaac Gym)"""
        n = self.config.num_joints

        # State tensors
        self.base_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.base_pos[:, 2] = 1.0  # Standing height

        self.base_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self.base_quat[:, 0] = 1.0  # Identity quaternion

        self.base_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.base_ang_vel = torch.zeros(self.num_envs, 3, device=self.device)

        self.dof_pos = torch.zeros(self.num_envs, n, device=self.device)
        self.dof_vel = torch.zeros(self.num_envs, n, device=self.device)

        self.torques = torch.zeros(self.num_envs, n, device=self.device)

    def _create_ground_plane(self):
        """Create ground plane"""
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        plane_params.static_friction = self.config.randomize_friction[0]
        plane_params.dynamic_friction = self.config.randomize_friction[0]
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self):
        """Create environments and actors"""
        # Load robot asset
        asset_root = "."
        asset_file = self.config.robot_asset

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        asset_options.collapse_fixed_joints = True
        asset_options.replace_cylinder_with_capsule = True
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = False

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        # Create environments
        env_lower = gymapi.Vec3(-self.config.env_spacing, -self.config.env_spacing, 0.0)
        env_upper = gymapi.Vec3(self.config.env_spacing, self.config.env_spacing, self.config.env_spacing)

        self.envs = []
        self.actor_handles = []

        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))

            # Create robot actor
            start_pose = gymapi.Transform()
            start_pose.p = gymapi.Vec3(0, 0, 1.0)
            start_pose.r = gymapi.Quat(0, 0, 0, 1)

            actor_handle = self.gym.create_actor(env, robot_asset, start_pose, "robot", i, 0)
            self.envs.append(env)
            self.actor_handles.append(actor_handle)

    def _create_tensors(self):
        """Create GPU tensors for fast access"""
        # Get tensor pointers
        self.root_states = gymtorch.wrap_tensor(
            self.gym.acquire_actor_root_state_tensor(self.sim)
        )
        self.dof_state = gymtorch.wrap_tensor(
            self.gym.acquire_dof_state_tensor(self.sim)
        )

        # Parse tensors
        self.base_pos = self.root_states[:, 0:3]
        self.base_quat = self.root_states[:, 3:7]
        self.base_vel = self.root_states[:, 7:10]
        self.base_ang_vel = self.root_states[:, 10:13]

        self.dof_pos = self.dof_state[:, 0::2]
        self.dof_vel = self.dof_state[:, 1::2]

    def reset(self, env_ids: torch.Tensor = None) -> torch.Tensor:
        """
        Reset environments

        Args:
            env_ids: Environment indices to reset (None = all)

        Returns:
            Initial observations
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # Reset positions
        self.base_pos[env_ids, :2] = 0.0
        self.base_pos[env_ids, 2] = 1.0

        self.base_quat[env_ids] = torch.tensor([1, 0, 0, 0], device=self.device).float()
        self.base_vel[env_ids] = 0.0
        self.base_ang_vel[env_ids] = 0.0

        self.dof_pos[env_ids] = self.default_dof_pos
        self.dof_vel[env_ids] = 0.0

        # Sample new commands
        self._sample_commands(env_ids)

        # Reset buffers
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = False
        self.last_actions[env_ids] = 0.0

        return self._compute_observations()

    def _sample_commands(self, env_ids: torch.Tensor):
        """Sample new velocity commands"""
        n = len(env_ids)

        # Sample forward velocity
        self.commands[env_ids, 0] = torch.rand(n, device=self.device) * 1.0  # 0-1 m/s

        # Sample lateral velocity (smaller range)
        self.commands[env_ids, 1] = (torch.rand(n, device=self.device) - 0.5) * 0.5

        # Sample yaw rate
        self.commands[env_ids, 2] = (torch.rand(n, device=self.device) - 0.5) * 1.0

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Step environment

        Args:
            actions: Actions (num_envs, num_actions)

        Returns:
            Tuple of (obs, rewards, dones, info)
        """
        # Clip actions
        actions = torch.clamp(actions, -self.config.clip_actions, self.config.clip_actions)

        # Scale actions
        scaled_actions = actions * self.config.action_scale

        # Convert to torques (PD control)
        torques = self._compute_torques(scaled_actions)

        if ISAACGYM_AVAILABLE:
            # Apply torques
            self.gym.set_dof_actuation_force_tensor(
                self.sim,
                gymtorch.unwrap_tensor(torques)
            )

            # Step simulation
            n_steps = int(self.config.control_dt / self.config.sim_dt)
            for _ in range(n_steps):
                self.gym.simulate(self.sim)

            # Refresh tensors
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)
        else:
            # Simple integration
            self._simple_step(torques)

        # Compute observations
        obs = self._compute_observations()

        # Compute rewards
        rewards = self._compute_rewards(actions)

        # Check terminations
        dones = self._check_terminations()

        # Update episode length
        self.episode_length_buf += 1

        # Store last actions
        self.last_actions = actions.clone()

        # Reset terminated environments
        reset_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_ids) > 0:
            self.reset(reset_ids)

        info = {
            'episode_length': self.episode_length_buf.clone()
        }

        return obs, rewards, dones, info

    def _simple_step(self, torques: torch.Tensor):
        """Simple physics step (no Isaac Gym)"""
        dt = self.config.control_dt

        # Joint dynamics
        dof_acc = (torques - 10.0 * self.dof_pos - 2.0 * self.dof_vel) / 5.0
        self.dof_vel += dof_acc * dt
        self.dof_pos += self.dof_vel * dt

        # Base dynamics (simplified)
        gravity = torch.tensor([0, 0, -9.81], device=self.device)
        self.base_vel += gravity * dt
        self.base_pos += self.base_vel * dt

        # Ground contact
        below_ground = self.base_pos[:, 2] < 0.3
        self.base_pos[below_ground, 2] = 0.3
        self.base_vel[below_ground, 2] = 0.0

    def _compute_torques(self, actions: torch.Tensor) -> torch.Tensor:
        """Compute joint torques from actions (PD control)"""
        # Actions are target positions
        target_pos = self.default_dof_pos + actions

        # PD control
        kp = 50.0
        kd = 2.0

        torques = kp * (target_pos - self.dof_pos) - kd * self.dof_vel

        # Clamp torques
        torques = torch.clamp(torques, -100.0, 100.0)

        return torques

    def _compute_observations(self) -> torch.Tensor:
        """Compute observations"""
        scales = self.config.obs_scales

        # Base state (in body frame)
        base_euler = self._quat_to_euler(self.base_quat)

        obs_list = [
            self.base_pos * scales['base_pos'],
            base_euler * scales['base_pos'],
            self.base_vel * scales['base_vel'],
            self.base_ang_vel * scales['base_ang_vel'],
            self.dof_pos * scales['joint_pos'],
            self.dof_vel * scales['joint_vel'],
            self.commands * scales['command'],
            self.last_actions
        ]

        return torch.cat(obs_list, dim=-1)

    def _compute_rewards(self, actions: torch.Tensor) -> torch.Tensor:
        """Compute rewards"""
        scales = self.config.reward_scales

        # Velocity tracking
        vel_error = torch.sum(torch.square(
            self.base_vel[:, :2] - self.commands[:, :2]
        ), dim=1)
        rew_vel = torch.exp(-vel_error / 0.25) * scales['tracking_linear_vel']

        # Yaw rate tracking
        yaw_error = torch.square(self.base_ang_vel[:, 2] - self.commands[:, 2])
        rew_yaw = torch.exp(-yaw_error / 0.25) * scales['tracking_angular_vel']

        # Height penalty
        height_error = torch.square(self.base_pos[:, 2] - 1.0)
        rew_height = height_error * scales['base_height']

        # Torque penalty
        rew_torque = torch.sum(torch.square(self.torques), dim=1) * scales['torques']

        # Action rate penalty
        action_rate = torch.sum(torch.square(actions - self.last_actions), dim=1)
        rew_action_rate = action_rate * scales['action_rate']

        # Alive bonus
        rew_alive = torch.ones(self.num_envs, device=self.device) * scales['alive']

        total_reward = rew_vel + rew_yaw + rew_height + rew_torque + rew_action_rate + rew_alive

        return total_reward

    def _check_terminations(self) -> torch.Tensor:
        """Check for episode termination"""
        # Height termination
        height_term = self.base_pos[:, 2] < self.config.termination_height

        # Max episode length
        max_length = 1000
        time_term = self.episode_length_buf >= max_length

        return height_term | time_term

    def _quat_to_euler(self, quat: torch.Tensor) -> torch.Tensor:
        """Convert quaternion to Euler angles"""
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

        # Roll
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)

        # Pitch
        sinp = 2 * (w * y - z * x)
        sinp = torch.clamp(sinp, -1, 1)
        pitch = torch.asin(sinp)

        # Yaw
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = torch.atan2(siny_cosp, cosy_cosp)

        return torch.stack([roll, pitch, yaw], dim=-1)

    def close(self):
        """Close environment"""
        if ISAACGYM_AVAILABLE:
            self.gym.destroy_sim(self.sim)
