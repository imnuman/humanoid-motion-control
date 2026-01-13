#!/usr/bin/env python3
"""
Centroidal Model Predictive Control for Humanoid Locomotion
Real-time optimization for contact force planning using convex QP

Author: Al Numan
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum
import time
import yaml
from pathlib import Path

from .mpc_formulation import ConvexMPCFormulation, MPCWeights
from .gait_scheduler import AdvancedGaitScheduler, GaitPattern


class ContactState(Enum):
    """Contact state for each foot"""
    STANCE = 1
    SWING = 0


@dataclass
class RobotParams:
    """Robot physical parameters"""
    mass: float = 47.0                    # kg
    gravity: float = 9.81                 # m/s^2
    standing_height: float = 1.0          # m
    foot_size: Tuple[float, float] = (0.15, 0.08)  # length, width
    max_force: float = 500.0              # N per foot
    friction_coefficient: float = 0.6
    hip_width: float = 0.2               # Distance between feet
    inertia: np.ndarray = None           # 3x3 rotational inertia

    def __post_init__(self):
        if self.inertia is None:
            # Approximate inertia for humanoid
            self.inertia = np.diag([5.0, 5.0, 1.0])


@dataclass
class MPCConfig:
    """MPC configuration parameters"""
    horizon: int = 10                     # prediction steps
    dt: float = 0.02                      # timestep (50 Hz)

    # State tracking weights [x, y, z]
    Q_pos: np.ndarray = None
    Q_vel: np.ndarray = None
    Q_ori: np.ndarray = None
    Q_ang_vel: np.ndarray = None

    # Control weights
    R_force: float = 1e-5
    R_force_rate: float = 1e-4

    def __post_init__(self):
        if self.Q_pos is None:
            self.Q_pos = np.array([10.0, 10.0, 100.0])
        if self.Q_vel is None:
            self.Q_vel = np.array([2.0, 2.0, 10.0])
        if self.Q_ori is None:
            self.Q_ori = np.array([50.0, 50.0, 10.0])
        if self.Q_ang_vel is None:
            self.Q_ang_vel = np.array([1.0, 1.0, 5.0])

    @classmethod
    def from_yaml(cls, filepath: str) -> 'MPCConfig':
        """Load configuration from YAML file"""
        with open(filepath, 'r') as f:
            cfg = yaml.safe_load(f)

        mpc_cfg = cfg.get('mpc', {})
        weights = mpc_cfg.get('weights', {})

        return cls(
            horizon=mpc_cfg.get('horizon', 10),
            dt=mpc_cfg.get('dt', 0.02),
            Q_pos=np.array(weights.get('position', [10.0, 10.0, 100.0])),
            Q_vel=np.array(weights.get('velocity', [2.0, 2.0, 10.0])),
            Q_ori=np.array(weights.get('orientation', [50.0, 50.0, 10.0])),
            Q_ang_vel=np.array(weights.get('angular_velocity', [1.0, 1.0, 5.0])),
            R_force=weights.get('force', 1e-5),
            R_force_rate=weights.get('force_rate', 1e-4)
        )


@dataclass
class State:
    """Robot centroidal state"""
    position: np.ndarray       # (3,) CoM position [x, y, z]
    velocity: np.ndarray       # (3,) CoM velocity
    orientation: np.ndarray    # (3,) Euler angles [roll, pitch, yaw]
    angular_velocity: np.ndarray  # (3,) body angular velocity

    @classmethod
    def zero(cls) -> 'State':
        """Create zero state at nominal height"""
        return cls(
            position=np.array([0.0, 0.0, 1.0]),
            velocity=np.zeros(3),
            orientation=np.zeros(3),
            angular_velocity=np.zeros(3)
        )

    def to_vector(self) -> np.ndarray:
        """Convert to 12D state vector"""
        return np.concatenate([
            self.position,
            self.orientation,
            self.velocity,
            self.angular_velocity
        ])

    @classmethod
    def from_vector(cls, vec: np.ndarray) -> 'State':
        """Create state from 12D vector"""
        return cls(
            position=vec[0:3].copy(),
            orientation=vec[3:6].copy(),
            velocity=vec[6:9].copy(),
            angular_velocity=vec[9:12].copy()
        )

    def copy(self) -> 'State':
        """Create a copy of this state"""
        return State(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            orientation=self.orientation.copy(),
            angular_velocity=self.angular_velocity.copy()
        )


@dataclass
class MPCResult:
    """MPC solution result"""
    left_force: np.ndarray
    right_force: np.ndarray
    solve_time_ms: float
    status: str
    predicted_trajectory: Optional[List[State]] = None


class CentroidalMPC:
    """
    Centroidal Model Predictive Control for Humanoid Robots

    Optimizes contact forces to track desired CoM motion using
    simplified centroidal dynamics and convex optimization.

    Features:
    - Real-time convex QP solving with OSQP
    - Multiple gait patterns support
    - Friction cone constraints
    - Warm starting for faster solve times
    """

    def __init__(
        self,
        robot_params: RobotParams = None,
        config: MPCConfig = None,
        gait: GaitPattern = GaitPattern.WALK
    ):
        """
        Initialize Centroidal MPC

        Args:
            robot_params: Robot physical parameters
            config: MPC configuration
            gait: Initial gait pattern
        """
        self.robot = robot_params or RobotParams()
        self.config = config or MPCConfig()

        # Initialize gait scheduler
        self.gait = AdvancedGaitScheduler(gait=gait, dt=self.config.dt)

        # Initialize QP solver
        self.qp_solver = ConvexMPCFormulation(
            mass=self.robot.mass,
            gravity=self.robot.gravity,
            inertia=self.robot.inertia,
            horizon=self.config.horizon,
            dt=self.config.dt,
            weights=MPCWeights(
                position=self.config.Q_pos,
                velocity=self.config.Q_vel,
                orientation=self.config.Q_ori,
                angular_velocity=self.config.Q_ang_vel,
                force=self.config.R_force
            )
        )

        # State dimensions
        self.nx = 12  # [pos(3), ori(3), vel(3), ang_vel(3)]
        self.nu = 6   # [f_left(3), f_right(3)]

        # Velocity commands
        self.cmd_vel_x = 0.0
        self.cmd_vel_y = 0.0
        self.cmd_yaw_rate = 0.0

        # Foot positions relative to CoM (body frame)
        half_width = self.robot.hip_width / 2
        self.nominal_foot_positions = {
            'left': np.array([0.0, half_width, -self.robot.standing_height]),
            'right': np.array([0.0, -half_width, -self.robot.standing_height])
        }

        # Current foot positions (world frame)
        self.foot_positions = {
            'left': np.array([0.0, half_width, 0.0]),
            'right': np.array([0.0, -half_width, 0.0])
        }

        # Previous solution for warm starting
        self.prev_solution = None

        # Statistics
        self.solve_count = 0
        self.total_solve_time = 0.0

    def set_command(
        self,
        vx: float = 0.0,
        vy: float = 0.0,
        yaw_rate: float = 0.0
    ):
        """
        Set velocity command

        Args:
            vx: Forward velocity (m/s)
            vy: Lateral velocity (m/s)
            yaw_rate: Yaw rate (rad/s)
        """
        self.cmd_vel_x = vx
        self.cmd_vel_y = vy
        self.cmd_yaw_rate = yaw_rate

    def set_gait(self, gait: GaitPattern, immediate: bool = False):
        """
        Set gait pattern

        Args:
            gait: Target gait pattern
            immediate: If True, switch immediately
        """
        self.gait.set_gait(gait, immediate)

    def update_foot_positions(self, positions: Dict[str, np.ndarray]):
        """Update current foot positions in world frame"""
        self.foot_positions = {
            k: v.copy() for k, v in positions.items()
        }

    def get_reference_trajectory(
        self,
        current_state: State
    ) -> List[np.ndarray]:
        """
        Generate reference trajectory from velocity command

        Args:
            current_state: Current robot state

        Returns:
            List of reference state vectors for MPC horizon
        """
        trajectory = []
        pos = current_state.position.copy()
        ori = current_state.orientation.copy()
        vel = np.array([self.cmd_vel_x, self.cmd_vel_y, 0.0])
        ang_vel = np.array([0.0, 0.0, self.cmd_yaw_rate])

        for k in range(self.config.horizon):
            # Propagate position
            pos = pos + vel * self.config.dt
            pos[2] = self.robot.standing_height  # Maintain height

            # Propagate orientation
            ori = ori + ang_vel * self.config.dt

            # Build reference state vector
            x_ref = np.concatenate([pos, ori, vel, ang_vel])
            trajectory.append(x_ref)

        return trajectory

    def get_contact_positions_relative(
        self,
        current_state: State
    ) -> List[np.ndarray]:
        """
        Get contact positions relative to CoM in world frame

        Args:
            current_state: Current robot state

        Returns:
            List of contact positions [left, right]
        """
        com = current_state.position
        return [
            self.foot_positions['left'] - com,
            self.foot_positions['right'] - com
        ]

    def solve(
        self,
        current_state: State,
        foot_positions: Optional[Dict[str, np.ndarray]] = None
    ) -> MPCResult:
        """
        Solve MPC optimization

        Args:
            current_state: Current robot centroidal state
            foot_positions: Optional foot positions in world frame

        Returns:
            MPCResult with optimal forces
        """
        start_time = time.time()

        # Update foot positions if provided
        if foot_positions:
            self.update_foot_positions(foot_positions)

        # Get reference trajectory
        x_ref = self.get_reference_trajectory(current_state)

        # Get contact schedule
        contact_schedule_raw = self.gait.get_contact_schedule(
            self.config.horizon, self.config.dt
        )

        # Convert to list of lists for solver
        contact_schedule = [
            [cs['left'], cs['right']]
            for cs in contact_schedule_raw
        ]

        # Get contact positions relative to CoM
        contact_positions = self.get_contact_positions_relative(current_state)

        # Current state vector
        x0 = current_state.to_vector()

        # Solve QP
        result = self.qp_solver.solve(
            x0=x0,
            x_ref=x_ref,
            contact_positions=contact_positions,
            contact_schedule=contact_schedule,
            warm_start=self.prev_solution
        )

        # Store solution for warm starting
        if result['full_solution'] is not None:
            self.prev_solution = result['full_solution']

        solve_time = (time.time() - start_time) * 1000

        # Update statistics
        self.solve_count += 1
        self.total_solve_time += solve_time

        # Extract forces
        forces = result['forces']
        left_force = forces[0:3]
        right_force = forces[3:6]

        # Apply contact mask
        if not contact_schedule[0][0]:  # Left foot swing
            left_force = np.zeros(3)
        if not contact_schedule[0][1]:  # Right foot swing
            right_force = np.zeros(3)

        return MPCResult(
            left_force=left_force,
            right_force=right_force,
            solve_time_ms=solve_time,
            status=result['status']
        )

    def update(self, dt: Optional[float] = None):
        """
        Update internal state (gait phase)

        Args:
            dt: Time step (uses config dt if None)
        """
        self.gait.update(dt or self.config.dt)

    def get_statistics(self) -> Dict:
        """Get solver statistics"""
        return {
            'solve_count': self.solve_count,
            'total_solve_time_ms': self.total_solve_time,
            'avg_solve_time_ms': (
                self.total_solve_time / self.solve_count
                if self.solve_count > 0 else 0.0
            )
        }

    def reset(self):
        """Reset MPC state"""
        self.prev_solution = None
        self.solve_count = 0
        self.total_solve_time = 0.0
        self.gait = AdvancedGaitScheduler(
            gait=self.gait.current_gait,
            dt=self.config.dt
        )


def demo():
    """Demonstration of Centroidal MPC"""
    print("=" * 60)
    print("Centroidal MPC Demo - Humanoid Locomotion")
    print("=" * 60)

    # Create MPC controller
    robot = RobotParams(mass=47.0, standing_height=1.0)
    config = MPCConfig(horizon=10, dt=0.02)
    mpc = CentroidalMPC(robot_params=robot, config=config)

    # Set walking command
    mpc.set_command(vx=0.5, vy=0.0, yaw_rate=0.1)

    # Initial state
    state = State(
        position=np.array([0.0, 0.0, 1.0]),
        velocity=np.array([0.0, 0.0, 0.0]),
        orientation=np.array([0.0, 0.0, 0.0]),
        angular_velocity=np.array([0.0, 0.0, 0.0])
    )

    print(f"\nRobot mass: {robot.mass} kg")
    print(f"Standing height: {robot.standing_height} m")
    print(f"MPC horizon: {config.horizon}")
    print(f"Timestep: {config.dt*1000:.0f} ms")
    print(f"\nCommand: vx={mpc.cmd_vel_x} m/s, yaw_rate={mpc.cmd_yaw_rate} rad/s")
    print("\n" + "-" * 60)

    # Simulation loop
    dt = 0.02
    for step in range(100):
        # Solve MPC
        result = mpc.solve(state)

        # Update gait
        mpc.update(dt)

        # Print results every 10 steps
        if step % 10 == 0:
            print(f"\nStep {step:3d}:")
            print(f"  Position: [{state.position[0]:.3f}, "
                  f"{state.position[1]:.3f}, {state.position[2]:.3f}]")
            print(f"  Velocity: [{state.velocity[0]:.3f}, "
                  f"{state.velocity[1]:.3f}, {state.velocity[2]:.3f}]")
            print(f"  Left force:  [{result.left_force[0]:6.1f}, "
                  f"{result.left_force[1]:6.1f}, {result.left_force[2]:6.1f}] N")
            print(f"  Right force: [{result.right_force[0]:6.1f}, "
                  f"{result.right_force[1]:6.1f}, {result.right_force[2]:6.1f}] N")
            print(f"  Solve time: {result.solve_time_ms:.2f} ms "
                  f"({result.status})")

        # Simple dynamics update (would use full dynamics in practice)
        total_force = result.left_force + result.right_force
        gravity = np.array([0, 0, -robot.gravity * robot.mass])
        net_force = total_force + gravity

        acceleration = net_force / robot.mass
        state.velocity = state.velocity + acceleration * dt
        state.position = state.position + state.velocity * dt

        # Keep at nominal height (simplified)
        state.position[2] = max(state.position[2], 0.9)

    # Print statistics
    stats = mpc.get_statistics()
    print("\n" + "=" * 60)
    print("Statistics:")
    print(f"  Total solves: {stats['solve_count']}")
    print(f"  Average solve time: {stats['avg_solve_time_ms']:.2f} ms")
    print("=" * 60)


if __name__ == "__main__":
    demo()
