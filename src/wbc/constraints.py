#!/usr/bin/env python3
"""
Constraint definitions for Whole-Body Control
Friction cones, torque limits, and joint limits
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional


class Constraint(ABC):
    """Abstract base class for WBC constraints"""

    def __init__(self, name: str):
        self.name = name
        self.active = True

    @abstractmethod
    def compute(
        self,
        q: np.ndarray,
        v: np.ndarray,
        robot_model
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute constraint matrices

        Returns:
            Tuple of (A, lb, ub) for constraint lb <= A*x <= ub
        """
        pass


@dataclass
class FrictionConeConstraint(Constraint):
    """
    Friction cone constraint for contact forces

    Approximates friction cone with 4-sided pyramid:
    |fx| <= mu * fz
    |fy| <= mu * fz
    fz >= fmin
    """

    def __init__(
        self,
        name: str,
        contact_frame: str,
        friction_coefficient: float = 0.6,
        min_normal_force: float = 10.0,
        max_normal_force: float = 500.0
    ):
        super().__init__(name)
        self.contact_frame = contact_frame
        self.mu = friction_coefficient
        self.f_min = min_normal_force
        self.f_max = max_normal_force

    def compute(
        self,
        q: np.ndarray,
        v: np.ndarray,
        robot_model
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute friction cone constraint matrices

        Decision variable ordering assumed: [ddq, f_contacts, tau]
        This returns constraint on contact forces only
        """
        # 6 constraints per contact:
        # fx + mu*fz >= 0
        # -fx + mu*fz >= 0
        # fy + mu*fz >= 0
        # -fy + mu*fz >= 0
        # fz >= fmin
        # fz <= fmax

        n_constraints = 6
        n_force_vars = 3  # fx, fy, fz

        A = np.zeros((n_constraints, n_force_vars))
        lb = np.zeros(n_constraints)
        ub = np.full(n_constraints, np.inf)

        # fx + mu*fz >= 0
        A[0, 0] = 1.0
        A[0, 2] = self.mu
        lb[0] = 0.0

        # -fx + mu*fz >= 0
        A[1, 0] = -1.0
        A[1, 2] = self.mu
        lb[1] = 0.0

        # fy + mu*fz >= 0
        A[2, 1] = 1.0
        A[2, 2] = self.mu
        lb[2] = 0.0

        # -fy + mu*fz >= 0
        A[3, 1] = -1.0
        A[3, 2] = self.mu
        lb[3] = 0.0

        # fmin <= fz <= fmax
        A[4, 2] = 1.0
        lb[4] = self.f_min
        ub[4] = self.f_max

        A[5, 2] = -1.0
        lb[5] = -self.f_max
        ub[5] = -self.f_min

        return A, lb, ub

    def get_friction_cone_matrix(self) -> np.ndarray:
        """
        Get friction cone constraint matrix in standard form
        Cf @ f >= 0
        """
        mu = self.mu
        Cf = np.array([
            [1, 0, mu],    # fx + mu*fz >= 0
            [-1, 0, mu],   # -fx + mu*fz >= 0
            [0, 1, mu],    # fy + mu*fz >= 0
            [0, -1, mu],   # -fy + mu*fz >= 0
            [0, 0, 1]      # fz >= 0
        ])
        return Cf


@dataclass
class TorqueLimitConstraint(Constraint):
    """Joint torque limit constraint"""

    def __init__(
        self,
        name: str = "torque_limits",
        torque_limits: np.ndarray = None,
        num_joints: int = 19
    ):
        super().__init__(name)
        if torque_limits is None:
            self.torque_limits = np.full(num_joints, 100.0)
        else:
            self.torque_limits = torque_limits
        self.num_joints = num_joints

    def compute(
        self,
        q: np.ndarray,
        v: np.ndarray,
        robot_model
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute torque limit constraints"""
        n = self.num_joints

        # -tau_max <= tau <= tau_max
        A = np.eye(n)
        lb = -self.torque_limits
        ub = self.torque_limits

        return A, lb, ub


@dataclass
class JointLimitConstraint(Constraint):
    """Joint position and velocity limit constraint"""

    def __init__(
        self,
        name: str = "joint_limits",
        position_lower: np.ndarray = None,
        position_upper: np.ndarray = None,
        velocity_limits: np.ndarray = None,
        num_joints: int = 19,
        dt: float = 0.002
    ):
        super().__init__(name)
        self.num_joints = num_joints
        self.dt = dt

        if position_lower is None:
            self.position_lower = np.full(num_joints, -np.pi)
        else:
            self.position_lower = position_lower

        if position_upper is None:
            self.position_upper = np.full(num_joints, np.pi)
        else:
            self.position_upper = position_upper

        if velocity_limits is None:
            self.velocity_limits = np.full(num_joints, 10.0)
        else:
            self.velocity_limits = velocity_limits

    def compute(
        self,
        q: np.ndarray,
        v: np.ndarray,
        robot_model
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute joint limit constraints

        Using barrier approach:
        ddq such that q + v*dt + 0.5*ddq*dt^2 stays within limits
        """
        n = self.num_joints

        # Extract joint positions and velocities
        q_joints = q[7:7+n] if len(q) > n else q
        v_joints = v[6:6+n] if len(v) > n else v

        dt = self.dt

        # Position limits as acceleration constraints
        # q_next = q + v*dt + 0.5*ddq*dt^2
        # q_lower <= q_next <= q_upper
        # => (q_lower - q - v*dt) / (0.5*dt^2) <= ddq <= (q_upper - q - v*dt) / (0.5*dt^2)

        coeff = 0.5 * dt * dt

        # Position constraint on accelerations
        A_pos = np.eye(n)
        lb_pos = (self.position_lower - q_joints - v_joints * dt) / coeff
        ub_pos = (self.position_upper - q_joints - v_joints * dt) / coeff

        # Velocity limits
        # v_next = v + ddq * dt
        # -v_max <= v_next <= v_max
        # => (-v_max - v) / dt <= ddq <= (v_max - v) / dt

        A_vel = np.eye(n)
        lb_vel = (-self.velocity_limits - v_joints) / dt
        ub_vel = (self.velocity_limits - v_joints) / dt

        # Combine constraints
        A = np.vstack([A_pos, A_vel])
        lb = np.concatenate([lb_pos, lb_vel])
        ub = np.concatenate([ub_pos, ub_vel])

        return A, lb, ub


class Constraints:
    """Container for all WBC constraints"""

    def __init__(self):
        self.friction_cones: List[FrictionConeConstraint] = []
        self.torque_limits: Optional[TorqueLimitConstraint] = None
        self.joint_limits: Optional[JointLimitConstraint] = None

    def add_friction_cone(
        self,
        contact_frame: str,
        friction_coefficient: float = 0.6,
        min_normal_force: float = 10.0,
        max_normal_force: float = 500.0
    ):
        """Add friction cone constraint for a contact"""
        constraint = FrictionConeConstraint(
            name=f"friction_{contact_frame}",
            contact_frame=contact_frame,
            friction_coefficient=friction_coefficient,
            min_normal_force=min_normal_force,
            max_normal_force=max_normal_force
        )
        self.friction_cones.append(constraint)

    def set_torque_limits(
        self,
        torque_limits: np.ndarray,
        num_joints: int = 19
    ):
        """Set joint torque limits"""
        self.torque_limits = TorqueLimitConstraint(
            name="torque_limits",
            torque_limits=torque_limits,
            num_joints=num_joints
        )

    def set_joint_limits(
        self,
        position_lower: np.ndarray,
        position_upper: np.ndarray,
        velocity_limits: np.ndarray,
        num_joints: int = 19,
        dt: float = 0.002
    ):
        """Set joint position and velocity limits"""
        self.joint_limits = JointLimitConstraint(
            name="joint_limits",
            position_lower=position_lower,
            position_upper=position_upper,
            velocity_limits=velocity_limits,
            num_joints=num_joints,
            dt=dt
        )

    def build_constraint_matrices(
        self,
        q: np.ndarray,
        v: np.ndarray,
        robot_model,
        n_ddq: int,
        n_forces: int,
        n_tau: int,
        contact_indices: List[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build complete constraint matrix for WBC QP

        Decision variables: x = [ddq, f_contacts, tau]

        Args:
            q: Joint positions
            v: Joint velocities
            robot_model: Robot model
            n_ddq: Number of acceleration variables
            n_forces: Number of force variables
            n_tau: Number of torque variables
            contact_indices: Indices of active contacts

        Returns:
            Tuple of (A, lb, ub) for the full constraint
        """
        n_vars = n_ddq + n_forces + n_tau

        A_rows = []
        lb_list = []
        ub_list = []

        # Friction cone constraints (on force variables)
        if contact_indices is None:
            contact_indices = list(range(len(self.friction_cones)))

        for idx in contact_indices:
            if idx < len(self.friction_cones):
                fc = self.friction_cones[idx]
                A_fc, lb_fc, ub_fc = fc.compute(q, v, robot_model)

                # Expand to full variable space
                A_full = np.zeros((A_fc.shape[0], n_vars))
                # Force variables start after ddq
                force_start = n_ddq + idx * 3
                A_full[:, force_start:force_start+3] = A_fc

                A_rows.append(A_full)
                lb_list.append(lb_fc)
                ub_list.append(ub_fc)

        # Torque limits (on tau variables)
        if self.torque_limits is not None:
            A_tau, lb_tau, ub_tau = self.torque_limits.compute(q, v, robot_model)

            A_full = np.zeros((A_tau.shape[0], n_vars))
            tau_start = n_ddq + n_forces
            A_full[:, tau_start:tau_start+n_tau] = A_tau

            A_rows.append(A_full)
            lb_list.append(lb_tau)
            ub_list.append(ub_tau)

        # Joint limits (on ddq variables)
        if self.joint_limits is not None:
            A_jl, lb_jl, ub_jl = self.joint_limits.compute(q, v, robot_model)

            A_full = np.zeros((A_jl.shape[0], n_vars))
            # Joint accelerations are in ddq after floating base
            A_full[:, 6:6+self.joint_limits.num_joints] = A_jl[:, :self.joint_limits.num_joints]

            A_rows.append(A_full)
            lb_list.append(lb_jl)
            ub_list.append(ub_jl)

        # Stack all constraints
        if A_rows:
            A = np.vstack(A_rows)
            lb = np.concatenate(lb_list)
            ub = np.concatenate(ub_list)
        else:
            A = np.zeros((1, n_vars))
            lb = np.array([-np.inf])
            ub = np.array([np.inf])

        return A, lb, ub
