#!/usr/bin/env python3
"""
Centroidal Model Predictive Control for Humanoid Locomotion
Real-time optimization for contact force planning
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum
import time


class ContactState(Enum):
    """Contact state for each foot"""
    STANCE = 1
    SWING = 0


@dataclass
class RobotParams:
    """Robot physical parameters"""
    mass: float = 47.0  # kg
    gravity: float = 9.81  # m/s^2
    standing_height: float = 1.0  # m
    foot_size: Tuple[float, float] = (0.15, 0.08)  # length, width
    max_force: float = 500.0  # N per foot
    friction_coefficient: float = 0.6


@dataclass
class MPCConfig:
    """MPC configuration"""
    horizon: int = 10  # prediction steps
    dt: float = 0.02  # timestep (50 Hz)

    # State weights [x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]
    Q_pos: np.ndarray = None
    Q_vel: np.ndarray = None
    Q_ori: np.ndarray = None

    # Control weights
    R_force: float = 1e-5
    R_force_rate: float = 1e-4

    def __post_init__(self):
        if self.Q_pos is None:
            self.Q_pos = np.array([10.0, 10.0, 100.0])
        if self.Q_vel is None:
            self.Q_vel = np.array([1.0, 1.0, 1.0])
        if self.Q_ori is None:
            self.Q_ori = np.array([100.0, 100.0, 10.0])  # roll, pitch, yaw


@dataclass
class State:
    """Robot centroidal state"""
    position: np.ndarray  # (3,) CoM position
    velocity: np.ndarray  # (3,) CoM velocity
    orientation: np.ndarray  # (3,) roll, pitch, yaw
    angular_velocity: np.ndarray  # (3,) body angular velocity

    @classmethod
    def zero(cls) -> 'State':
        return cls(
            position=np.zeros(3),
            velocity=np.zeros(3),
            orientation=np.zeros(3),
            angular_velocity=np.zeros(3)
        )

    def to_vector(self) -> np.ndarray:
        return np.concatenate([
            self.position, self.velocity,
            self.orientation, self.angular_velocity
        ])

    @classmethod
    def from_vector(cls, vec: np.ndarray) -> 'State':
        return cls(
            position=vec[0:3],
            velocity=vec[3:6],
            orientation=vec[6:9],
            angular_velocity=vec[9:12]
        )


class GaitScheduler:
    """Gait timing and contact schedule"""

    def __init__(
        self,
        stance_duration: float = 0.35,
        swing_duration: float = 0.25,
        phase_offset: float = 0.5  # For trot gait
    ):
        self.stance_duration = stance_duration
        self.swing_duration = swing_duration
        self.period = stance_duration + swing_duration
        self.phase_offset = phase_offset

        self.phase = 0.0  # Current gait phase [0, 1)

    def update(self, dt: float):
        """Update gait phase"""
        self.phase = (self.phase + dt / self.period) % 1.0

    def get_contact_schedule(self, horizon: int, dt: float) -> List[Dict[str, ContactState]]:
        """
        Get contact schedule for MPC horizon

        Returns:
            List of contact states for each timestep
        """
        schedule = []
        phase = self.phase

        for _ in range(horizon):
            # Left foot
            left_in_stance = phase < (self.stance_duration / self.period)

            # Right foot (offset by phase_offset)
            right_phase = (phase + self.phase_offset) % 1.0
            right_in_stance = right_phase < (self.stance_duration / self.period)

            schedule.append({
                'left': ContactState.STANCE if left_in_stance else ContactState.SWING,
                'right': ContactState.STANCE if right_in_stance else ContactState.SWING
            })

            phase = (phase + dt / self.period) % 1.0

        return schedule


class CentroidalMPC:
    """
    Centroidal Model Predictive Control

    Solves for optimal contact forces to track desired CoM trajectory
    using simplified centroidal dynamics.
    """

    def __init__(
        self,
        robot_params: RobotParams = None,
        config: MPCConfig = None
    ):
        self.robot = robot_params or RobotParams()
        self.config = config or MPCConfig()

        self.gait = GaitScheduler()

        # State and control dimensions
        self.nx = 12  # [pos(3), vel(3), ori(3), ang_vel(3)]
        self.nu = 12  # [f_left(3), f_right(3), tau_left(3), tau_right(3)]

        # Desired command
        self.cmd_vel_x = 0.0
        self.cmd_vel_y = 0.0
        self.cmd_yaw_rate = 0.0

        # Foot positions (body frame)
        self.foot_positions = {
            'left': np.array([0.0, 0.1, 0.0]),
            'right': np.array([0.0, -0.1, 0.0])
        }

        # Previous solution for warm starting
        self.prev_forces = None

    def set_command(
        self,
        vx: float = 0.0,
        vy: float = 0.0,
        yaw_rate: float = 0.0
    ):
        """Set velocity command"""
        self.cmd_vel_x = vx
        self.cmd_vel_y = vy
        self.cmd_yaw_rate = yaw_rate

    def get_desired_trajectory(
        self,
        current_state: State,
        horizon: int
    ) -> List[State]:
        """Generate desired trajectory from current state and command"""
        trajectory = []
        state = State(
            position=current_state.position.copy(),
            velocity=np.array([self.cmd_vel_x, self.cmd_vel_y, 0.0]),
            orientation=current_state.orientation.copy(),
            angular_velocity=np.array([0.0, 0.0, self.cmd_yaw_rate])
        )

        for i in range(horizon):
            # Update position based on velocity
            state.position = state.position + state.velocity * self.config.dt
            state.position[2] = self.robot.standing_height  # Maintain height

            # Update orientation
            state.orientation[2] += self.cmd_yaw_rate * self.config.dt

            trajectory.append(State(
                position=state.position.copy(),
                velocity=state.velocity.copy(),
                orientation=state.orientation.copy(),
                angular_velocity=state.angular_velocity.copy()
            ))

        return trajectory

    def build_dynamics_matrices(
        self,
        contact_schedule: List[Dict[str, ContactState]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build discrete-time dynamics matrices A, B

        Centroidal dynamics:
        m * ddot_p = sum(f_i) + m * g
        I * dot_omega = sum(r_i x f_i)
        """
        dt = self.config.dt
        m = self.robot.mass
        g = np.array([0, 0, -self.robot.gravity])

        # Simplified inertia (assuming uniform density box)
        I = np.diag([5.0, 5.0, 1.0])  # Approximate humanoid inertia
        I_inv = np.linalg.inv(I)

        # State transition matrix (linearized)
        A = np.eye(self.nx)
        A[0:3, 3:6] = np.eye(3) * dt  # position from velocity
        A[6:9, 9:12] = np.eye(3) * dt  # orientation from angular velocity

        # Control matrix
        B = np.zeros((self.nx, self.nu))

        # Velocity from force: dv = (1/m) * f * dt
        B[3:6, 0:3] = np.eye(3) * dt / m  # left foot force
        B[3:6, 3:6] = np.eye(3) * dt / m  # right foot force

        # Angular velocity from torque: domega = I^-1 * tau * dt
        B[9:12, 6:9] = I_inv * dt  # left foot torque
        B[9:12, 9:12] = I_inv * dt  # right foot torque

        return A, B

    def solve(
        self,
        current_state: State,
        foot_positions: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Solve MPC optimization problem

        Args:
            current_state: Current robot state
            foot_positions: Optional foot positions in world frame

        Returns:
            Dict with optimal forces for each foot
        """
        start_time = time.time()

        # Update foot positions if provided
        if foot_positions:
            self.foot_positions = foot_positions

        # Get contact schedule
        contact_schedule = self.gait.get_contact_schedule(
            self.config.horizon, self.config.dt
        )

        # Get desired trajectory
        desired_trajectory = self.get_desired_trajectory(
            current_state, self.config.horizon
        )

        # Build dynamics
        A, B = self.build_dynamics_matrices(contact_schedule)

        # Build QP problem
        # For simplicity, using a basic QP formulation
        # In practice, would use OSQP or similar solver

        H = self.config.horizon

        # Cost matrices
        Q = np.diag(np.concatenate([
            self.config.Q_pos,
            self.config.Q_vel,
            self.config.Q_ori,
            np.ones(3)  # angular velocity
        ]))

        R = np.eye(self.nu) * self.config.R_force

        # Solve using simple gradient descent (simplified)
        # Real implementation would use QP solver
        forces = self._solve_qp_simplified(
            current_state, desired_trajectory,
            contact_schedule, A, B, Q, R
        )

        solve_time = (time.time() - start_time) * 1000

        # Store for warm start
        self.prev_forces = forces

        return {
            'left_force': forces[0:3],
            'right_force': forces[3:6],
            'left_torque': forces[6:9],
            'right_torque': forces[9:12],
            'solve_time_ms': solve_time
        }

    def _solve_qp_simplified(
        self,
        current_state: State,
        desired_trajectory: List[State],
        contact_schedule: List[Dict[str, ContactState]],
        A: np.ndarray,
        B: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray
    ) -> np.ndarray:
        """
        Simplified QP solution using iterative method

        Real implementation would use OSQP, qpOASES, or similar
        """
        # Initial guess
        if self.prev_forces is not None:
            u = self.prev_forces.copy()
        else:
            # Start with gravity compensation
            f_gravity = self.robot.mass * self.robot.gravity / 2
            u = np.array([0, 0, f_gravity, 0, 0, f_gravity, 0, 0, 0, 0, 0, 0])

        # Simple gradient descent
        learning_rate = 0.01
        num_iters = 50

        x = current_state.to_vector()

        for _ in range(num_iters):
            # Predict state
            x_pred = A @ x + B @ u

            # Desired state at first step
            x_des = desired_trajectory[0].to_vector()

            # Cost gradient
            state_error = x_pred - x_des
            grad_state = B.T @ Q @ state_error
            grad_control = R @ u

            grad = grad_state + grad_control

            # Update
            u = u - learning_rate * grad

            # Apply contact constraints
            for i, (foot, contact) in enumerate([('left', 0), ('right', 3)]):
                if contact_schedule[0][foot] == ContactState.SWING:
                    u[i:i+3] = 0  # No force during swing
                else:
                    # Friction cone constraint
                    fz = u[i+2]
                    fz = max(fz, 10.0)  # Minimum normal force
                    fz = min(fz, self.robot.max_force)
                    u[i+2] = fz

                    fx_max = self.robot.friction_coefficient * fz
                    fy_max = self.robot.friction_coefficient * fz

                    u[i] = np.clip(u[i], -fx_max, fx_max)
                    u[i+1] = np.clip(u[i+1], -fy_max, fy_max)

        return u

    def update_gait(self, dt: float):
        """Update gait phase"""
        self.gait.update(dt)


class WholeBodyController:
    """
    Whole-Body Controller using Task-Space Inverse Dynamics

    Converts MPC contact forces to joint torques while satisfying constraints.
    """

    def __init__(
        self,
        robot_model: str = "h1",
        num_joints: int = 19
    ):
        self.robot_model = robot_model
        self.num_joints = num_joints

        # Task weights
        self.task_weights = {
            'com_tracking': 100.0,
            'swing_foot': 50.0,
            'torso_orientation': 30.0,
            'joint_regularization': 1.0
        }

        # Gains
        self.kp_com = np.array([100, 100, 100])
        self.kd_com = np.array([20, 20, 20])
        self.kp_foot = np.array([200, 200, 200])
        self.kd_foot = np.array([40, 40, 40])

    def solve(
        self,
        current_state: Dict,
        contact_forces: Dict[str, np.ndarray],
        desired_com: np.ndarray,
        desired_foot_poses: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Solve for joint torques

        Args:
            current_state: Current robot state (joint positions, velocities, etc.)
            contact_forces: Optimal contact forces from MPC
            desired_com: Desired CoM position
            desired_foot_poses: Desired swing foot poses

        Returns:
            Joint torques (num_joints,)
        """
        # Placeholder - would use Pinocchio for actual inverse dynamics
        torques = np.zeros(self.num_joints)

        # In practice:
        # 1. Compute task-space accelerations from PD control
        # 2. Stack tasks with priorities
        # 3. Solve constrained QP for accelerations
        # 4. Use inverse dynamics to get torques

        return torques


def main():
    """Demo of Centroidal MPC"""
    print("Centroidal MPC Demo")
    print("=" * 40)

    # Initialize MPC
    mpc = CentroidalMPC()
    mpc.set_command(vx=0.5, vy=0.0, yaw_rate=0.1)

    # Initial state
    state = State(
        position=np.array([0.0, 0.0, 1.0]),
        velocity=np.array([0.0, 0.0, 0.0]),
        orientation=np.array([0.0, 0.0, 0.0]),
        angular_velocity=np.array([0.0, 0.0, 0.0])
    )

    # Simulation
    dt = 0.02
    for i in range(100):
        # Solve MPC
        result = mpc.solve(state)

        # Update gait
        mpc.update_gait(dt)

        if i % 10 == 0:
            print(f"Step {i}:")
            print(f"  Left force: [{result['left_force'][0]:.1f}, "
                  f"{result['left_force'][1]:.1f}, {result['left_force'][2]:.1f}]")
            print(f"  Right force: [{result['right_force'][0]:.1f}, "
                  f"{result['right_force'][1]:.1f}, {result['right_force'][2]:.1f}]")
            print(f"  Solve time: {result['solve_time_ms']:.2f} ms")

        # Simple state update (would use full dynamics in practice)
        total_force = result['left_force'] + result['right_force']
        acceleration = total_force / mpc.robot.mass + np.array([0, 0, -mpc.robot.gravity])

        state.velocity = state.velocity + acceleration * dt
        state.position = state.position + state.velocity * dt


if __name__ == "__main__":
    main()
