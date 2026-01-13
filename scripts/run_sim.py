#!/usr/bin/env python3
"""
Main Simulation Runner for Humanoid Motion Control

Demonstrates the full control stack:
- Centroidal MPC for force planning
- Whole-Body Controller for torque computation
- State estimation
- MuJoCo simulation

Author: Al Numan
"""

import numpy as np
import argparse
import yaml
from pathlib import Path
import time
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from mpc import CentroidalMPC, MPCConfig, RobotParams, State, GaitPattern
from wbc import WholeBodyController, WBCConfig
from estimation import StateEstimator, EstimatorConfig, IMUData, JointState
from planning import SwingTrajectoryGenerator, FootstepPlanner, FootSide
from simulation import MuJoCoEnv, EnvConfig
from utils.robot_model import RobotModel


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class HumanoidController:
    """
    Complete humanoid control system

    Integrates:
    - MPC for contact force optimization
    - WBC for joint torque computation
    - State estimation
    - Swing trajectory planning
    """

    def __init__(self, config: dict):
        """
        Initialize controller

        Args:
            config: Configuration dictionary
        """
        self.config = config

        # Robot parameters
        robot_cfg = config.get('robot', {})
        self.robot_params = RobotParams(
            mass=robot_cfg.get('total_mass', 47.0),
            standing_height=robot_cfg.get('standing_height', 1.0),
            hip_width=robot_cfg.get('hip_width', 0.2)
        )

        # Robot model (simplified for now)
        self.robot_model = RobotModel()
        self.robot_model.config.total_mass = self.robot_params.mass
        self.robot_model.config.standing_height = self.robot_params.standing_height

        # MPC
        mpc_cfg = config.get('mpc', {})
        mpc_config = MPCConfig(
            horizon=mpc_cfg.get('horizon', 10),
            dt=mpc_cfg.get('dt', 0.02)
        )
        self.mpc = CentroidalMPC(
            robot_params=self.robot_params,
            config=mpc_config,
            gait=GaitPattern.WALK
        )

        # WBC
        wbc_cfg = config.get('wbc', {})
        wbc_config = WBCConfig(
            num_joints=robot_cfg.get('num_joints', 19),
            dt=wbc_cfg.get('dt', 0.002)
        )
        self.wbc = WholeBodyController(
            robot_model=self.robot_model,
            config=wbc_config
        )

        # State estimator
        est_cfg = config.get('estimation', {})
        est_config = EstimatorConfig(
            contact_frames=est_cfg.get('contact_frames', ['left_foot', 'right_foot'])
        )
        self.estimator = StateEstimator(
            robot_model=self.robot_model,
            config=est_config
        )

        # Swing trajectory generator
        gait_cfg = config.get('gait', {}).get('walk', {})
        self.swing_generator = SwingTrajectoryGenerator(
            trajectory_type="bezier",
            default_apex_height=gait_cfg.get('swing_height', 0.08),
            default_duration=gait_cfg.get('swing_duration', 0.25)
        )

        # Footstep planner
        self.footstep_planner = FootstepPlanner()

        # Velocity command
        self.cmd_vel = np.array([0.0, 0.0, 0.0])  # [vx, vy, yaw_rate]

        # Control rates
        self.mpc_rate = int(1.0 / mpc_config.dt)
        self.wbc_rate = int(1.0 / wbc_config.dt)
        self.mpc_counter = 0

        # State
        self.current_state = State.zero()

    def set_velocity_command(self, vx: float, vy: float, yaw_rate: float):
        """Set velocity command"""
        self.cmd_vel = np.array([vx, vy, yaw_rate])
        self.mpc.set_command(vx=vx, vy=vy, yaw_rate=yaw_rate)

    def update(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        imu_data: IMUData = None,
        contact_forces: dict = None,
        dt: float = 0.002
    ) -> np.ndarray:
        """
        Run control update

        Args:
            qpos: Joint positions
            qvel: Joint velocities
            imu_data: IMU sensor data
            contact_forces: Measured contact forces
            dt: Time step

        Returns:
            Joint torques
        """
        # Update state estimate
        if imu_data is not None:
            joint_state = JointState(
                positions=qpos[7:] if len(qpos) > 7 else qpos,
                velocities=qvel[6:] if len(qvel) > 6 else qvel
            )
            self.estimator.update(
                imu=imu_data,
                joint_state=joint_state,
                contact_state=self._get_contact_state(),
                dt=dt
            )

        # Update current state for MPC
        state_est = self.estimator.get_state()
        self.current_state = State(
            position=state_est.position,
            velocity=state_est.velocity,
            orientation=state_est.get_euler_orientation(),
            angular_velocity=state_est.angular_velocity
        )

        # Run MPC at lower rate
        self.mpc_counter += 1
        if self.mpc_counter >= self.wbc_rate // self.mpc_rate:
            self.mpc_counter = 0
            self._run_mpc()

        # Run WBC
        torques = self._run_wbc(qpos, qvel)

        # Update gait
        self.mpc.update(dt)

        return torques

    def _run_mpc(self):
        """Run MPC update"""
        # Get foot positions
        foot_positions = self.robot_model.get_foot_positions()

        # Solve MPC
        result = self.mpc.solve(self.current_state, foot_positions)

        # Update WBC with desired forces
        self.wbc.set_desired_forces({
            'left_foot': result.left_force,
            'right_foot': result.right_force
        })

    def _run_wbc(self, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        """Run WBC update"""
        # Update robot model
        self.robot_model.update_state(qpos, qvel)

        # Get contact state
        contact_state = self._get_contact_state()
        self.wbc.set_contact_state(contact_state)

        # Update task targets
        self._update_tasks()

        # Solve WBC
        result = self.wbc.solve(qpos, qvel)

        return result.joint_torques

    def _get_contact_state(self) -> dict:
        """Get current contact state from gait scheduler"""
        gait_state = self.mpc.gait.feet
        return {
            'left_foot': gait_state['left'].in_contact,
            'right_foot': gait_state['right'].in_contact
        }

    def _update_tasks(self):
        """Update WBC task targets"""
        # CoM target (track reference)
        com_target = np.array([
            self.current_state.position[0] + self.cmd_vel[0] * 0.1,
            self.current_state.position[1] + self.cmd_vel[1] * 0.1,
            self.robot_params.standing_height
        ])
        self.wbc.set_task_target('com_tracking', position=com_target)

        # Torso orientation (keep upright)
        self.wbc.set_task_target('torso_orientation',
                                orientation=np.array([0.0, 0.0, self.current_state.orientation[2]]))

        # Swing foot targets (from trajectory generator)
        gait_state = self.mpc.gait.feet
        for foot_name in ['left', 'right']:
            foot = gait_state[foot_name]
            if not foot.in_contact:
                # Generate swing trajectory
                start = foot.position
                end = foot.target_position
                traj = self.swing_generator.create_trajectory(start, end)

                # Get target pose
                t = foot.phase * traj.params.duration
                target_pos = traj.get_position(t)
                target_vel = traj.get_velocity(t)

                self.wbc.set_task_target(
                    f'{foot_name}_foot',
                    pose=np.concatenate([target_pos, np.zeros(3)]),
                    velocity=np.concatenate([target_vel, np.zeros(3)])
                )


def main():
    """Main simulation loop"""
    parser = argparse.ArgumentParser(description='Humanoid Motion Control Simulation')
    parser.add_argument('--robot', type=str, default='h1',
                       choices=['h1', 'gr1'], help='Robot to simulate')
    parser.add_argument('--render', action='store_true', help='Enable rendering')
    parser.add_argument('--duration', type=float, default=10.0,
                       help='Simulation duration (seconds)')
    parser.add_argument('--vx', type=float, default=0.3, help='Forward velocity')
    parser.add_argument('--vy', type=float, default=0.0, help='Lateral velocity')
    parser.add_argument('--yaw', type=float, default=0.0, help='Yaw rate')
    args = parser.parse_args()

    print("=" * 60)
    print("Humanoid Motion Control Simulation")
    print("=" * 60)

    # Load config
    config_path = Path(__file__).parent.parent / 'config' / f'{args.robot}_config.yaml'
    if config_path.exists():
        config = load_config(str(config_path))
        print(f"Loaded configuration: {config_path}")
    else:
        print(f"Config not found: {config_path}, using defaults")
        config = {}

    # Create simulation environment
    env_config = EnvConfig(
        timestep=0.001,
        control_dt=0.002,
        render=args.render
    )
    env = MuJoCoEnv(config=env_config)

    # Create controller
    controller = HumanoidController(config)
    controller.set_velocity_command(args.vx, args.vy, args.yaw)

    print(f"\nRobot: {args.robot}")
    print(f"Command: vx={args.vx:.2f} m/s, vy={args.vy:.2f} m/s, yaw={args.yaw:.2f} rad/s")
    print(f"Duration: {args.duration:.1f} s")
    print(f"Render: {args.render}")
    print("\n" + "-" * 60)

    # Reset simulation
    state = env.reset()

    # Statistics
    start_time = time.time()
    step_count = 0
    total_mpc_time = 0.0
    total_wbc_time = 0.0

    # Main loop
    n_steps = int(args.duration / env.dt)
    print_interval = n_steps // 20  # Print 20 times

    for step in range(n_steps):
        # Get sensor data
        sensors = env.get_sensor_data()

        # Create IMU data
        imu_data = IMUData(
            acceleration=sensors.get('imu_acceleration', np.array([0, 0, 9.81])),
            angular_velocity=sensors.get('imu_angular_velocity', np.zeros(3))
        )

        # Run controller
        mpc_start = time.time()
        torques = controller.update(
            qpos=state.qpos,
            qvel=state.qvel,
            imu_data=imu_data,
            dt=env.dt
        )
        mpc_end = time.time()
        total_mpc_time += (mpc_end - mpc_start) * 1000

        # Step simulation
        state = env.step(torques)

        # Render
        if args.render:
            env.render_frame()

        step_count += 1

        # Print progress
        if step % print_interval == 0:
            sim_time = state.time
            com_pos = state.qpos[:3]
            com_vel = state.qvel[:3]

            print(f"Time: {sim_time:6.2f}s | "
                  f"Pos: [{com_pos[0]:5.2f}, {com_pos[1]:5.2f}, {com_pos[2]:5.2f}] | "
                  f"Vel: [{com_vel[0]:5.2f}, {com_vel[1]:5.2f}]")

    # Final statistics
    wall_time = time.time() - start_time
    real_time_factor = args.duration / wall_time

    print("\n" + "=" * 60)
    print("Simulation Complete")
    print("=" * 60)
    print(f"Wall time: {wall_time:.2f} s")
    print(f"Sim time: {args.duration:.2f} s")
    print(f"Real-time factor: {real_time_factor:.2f}x")
    print(f"Steps: {step_count}")
    print(f"Avg control time: {total_mpc_time / step_count:.3f} ms/step")

    # Cleanup
    env.close()


if __name__ == "__main__":
    main()
