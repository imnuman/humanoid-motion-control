#!/usr/bin/env python3
"""
Whole-Body Controller for Humanoid Robots
Task-space inverse dynamics with hierarchical task composition

Author: Al Numan
"""

import numpy as np
from scipy import sparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import time

from .task import Task, TaskType, CoMTask, FootPoseTask, OrientationTask, JointRegularizationTask, TaskGains
from .constraints import Constraints, FrictionConeConstraint

try:
    import osqp
    OSQP_AVAILABLE = True
except ImportError:
    OSQP_AVAILABLE = False


@dataclass
class WBCConfig:
    """Whole-Body Controller configuration"""
    num_joints: int = 19           # Number of actuated joints
    dt: float = 0.002              # Control timestep (500 Hz)
    friction_coefficient: float = 0.6
    min_normal_force: float = 10.0
    max_normal_force: float = 500.0
    use_hierarchical: bool = False  # Use strict task hierarchy

    # Default task weights
    default_weights: Dict[str, float] = field(default_factory=lambda: {
        'com_tracking': 100.0,
        'swing_foot': 50.0,
        'torso_orientation': 30.0,
        'joint_regularization': 1.0
    })

    # Default PD gains
    default_com_gains: TaskGains = None
    default_foot_gains: TaskGains = None

    def __post_init__(self):
        if self.default_com_gains is None:
            self.default_com_gains = TaskGains(
                kp=np.array([100.0, 100.0, 100.0]),
                kd=np.array([20.0, 20.0, 20.0])
            )
        if self.default_foot_gains is None:
            self.default_foot_gains = TaskGains(
                kp=np.array([200.0, 200.0, 200.0, 100.0, 100.0, 100.0]),
                kd=np.array([40.0, 40.0, 40.0, 20.0, 20.0, 20.0])
            )


@dataclass
class WBCResult:
    """WBC solution result"""
    joint_torques: np.ndarray
    joint_accelerations: np.ndarray
    contact_forces: Dict[str, np.ndarray]
    solve_time_ms: float
    status: str


class WholeBodyController:
    """
    Whole-Body Controller using Task-Space Inverse Dynamics

    Converts MPC contact forces to joint torques while tracking
    multiple operational-space tasks and respecting constraints.

    The optimization problem:
    min  sum_i w_i * ||J_i * ddq - a_i_des||^2
    s.t. M * ddq + h = S' * tau + J_c' * f
         friction cone constraints on f
         torque limits on tau
         joint limits on ddq

    Decision variables: x = [ddq, f, tau]
    """

    def __init__(
        self,
        robot_model,
        config: WBCConfig = None
    ):
        """
        Initialize Whole-Body Controller

        Args:
            robot_model: Robot model for kinematics and dynamics
            config: WBC configuration
        """
        self.robot = robot_model
        self.config = config or WBCConfig()

        # Dimensions
        self.nq = robot_model.nq if hasattr(robot_model, 'nq') else self.config.num_joints + 7
        self.nv = robot_model.nv if hasattr(robot_model, 'nv') else self.config.num_joints + 6
        self.n_joints = self.config.num_joints

        # Tasks dictionary
        self.tasks: Dict[str, Task] = {}

        # Constraints
        self.constraints = Constraints()

        # Contact frames and state
        self.contact_frames = ['left_foot', 'right_foot']
        self.contact_state = {frame: False for frame in self.contact_frames}

        # Desired contact forces from MPC
        self.desired_forces = {frame: np.zeros(3) for frame in self.contact_frames}

        # Setup default tasks
        self._setup_default_tasks()

        # Setup constraints
        self._setup_constraints()

        # QP solver
        self.solver = None
        self.solver_initialized = False

        # Statistics
        self.solve_count = 0
        self.total_solve_time = 0.0

    def _setup_default_tasks(self):
        """Setup default tasks"""
        # CoM tracking
        self.add_task(CoMTask(
            name='com_tracking',
            weight=self.config.default_weights['com_tracking'],
            gains=self.config.default_com_gains
        ))

        # Left foot task
        self.add_task(FootPoseTask(
            name='left_foot',
            foot_frame='left_foot',
            weight=self.config.default_weights['swing_foot'],
            gains=self.config.default_foot_gains
        ))

        # Right foot task
        self.add_task(FootPoseTask(
            name='right_foot',
            foot_frame='right_foot',
            weight=self.config.default_weights['swing_foot'],
            gains=self.config.default_foot_gains
        ))

        # Torso orientation
        self.add_task(OrientationTask(
            name='torso_orientation',
            frame='base_link',
            weight=self.config.default_weights['torso_orientation']
        ))

        # Joint regularization (lowest priority)
        self.add_task(JointRegularizationTask(
            name='joint_regularization',
            num_joints=self.n_joints,
            weight=self.config.default_weights['joint_regularization']
        ))

    def _setup_constraints(self):
        """Setup default constraints"""
        # Friction cones for each contact
        for frame in self.contact_frames:
            self.constraints.add_friction_cone(
                contact_frame=frame,
                friction_coefficient=self.config.friction_coefficient,
                min_normal_force=self.config.min_normal_force,
                max_normal_force=self.config.max_normal_force
            )

        # Torque limits
        torque_limits = np.full(self.n_joints, 100.0)  # Default 100 Nm
        self.constraints.set_torque_limits(torque_limits, self.n_joints)

        # Joint limits
        self.constraints.set_joint_limits(
            position_lower=np.full(self.n_joints, -np.pi),
            position_upper=np.full(self.n_joints, np.pi),
            velocity_limits=np.full(self.n_joints, 10.0),
            num_joints=self.n_joints,
            dt=self.config.dt
        )

    def add_task(self, task: Task):
        """Add a task to the controller"""
        self.tasks[task.name] = task

    def remove_task(self, name: str):
        """Remove a task by name"""
        if name in self.tasks:
            del self.tasks[name]

    def set_task_weight(self, name: str, weight: float):
        """Set task weight"""
        if name in self.tasks:
            self.tasks[name].weight = weight

    def set_task_target(self, name: str, **kwargs):
        """Set task target"""
        if name in self.tasks:
            self.tasks[name].set_target(**kwargs)

    def set_contact_state(self, contact_state: Dict[str, bool]):
        """Set contact state for each foot"""
        self.contact_state = contact_state.copy()

        # Activate/deactivate foot tasks based on contact
        for frame, in_contact in contact_state.items():
            if frame in self.tasks:
                # High weight for stance foot (contact constraint)
                # Lower weight for swing foot (trajectory tracking)
                if in_contact:
                    self.tasks[frame].weight = 0.0  # Contact handled by dynamics
                else:
                    self.tasks[frame].weight = self.config.default_weights['swing_foot']

    def set_desired_forces(self, forces: Dict[str, np.ndarray]):
        """Set desired contact forces from MPC"""
        for frame, force in forces.items():
            if frame in self.desired_forces:
                self.desired_forces[frame] = force.copy()

    def solve(
        self,
        q: np.ndarray,
        v: np.ndarray,
        contact_forces: Optional[Dict[str, np.ndarray]] = None
    ) -> WBCResult:
        """
        Solve whole-body control optimization

        Args:
            q: Joint positions (nq,)
            v: Joint velocities (nv,)
            contact_forces: Desired contact forces from MPC

        Returns:
            WBCResult with joint torques and solution info
        """
        start_time = time.time()

        # Update robot state
        self.robot.update_state(q, v)

        # Update desired forces if provided
        if contact_forces:
            self.set_desired_forces(contact_forces)

        # Get active contacts
        active_contacts = [
            i for i, frame in enumerate(self.contact_frames)
            if self.contact_state.get(frame, False)
        ]
        n_contacts = len(active_contacts)
        n_forces = 3 * n_contacts

        # Decision variables: [ddq (nv), f (3*n_contacts), tau (n_joints)]
        n_ddq = self.nv
        n_tau = self.n_joints
        n_vars = n_ddq + n_forces + n_tau

        # Build QP problem
        P, q_vec = self._build_cost(q, v, n_ddq, n_forces, n_tau, active_contacts)
        A_eq, b_eq = self._build_dynamics_constraint(q, v, n_ddq, n_forces, n_tau, active_contacts)
        A_ineq, lb, ub = self._build_inequality_constraints(q, v, n_ddq, n_forces, n_tau, active_contacts)

        # Combine constraints
        A = np.vstack([A_eq, A_ineq])
        l = np.concatenate([b_eq, lb])
        u = np.concatenate([b_eq, ub])

        # Solve QP
        if OSQP_AVAILABLE:
            result = self._solve_qp(P, q_vec, A, l, u)
        else:
            result = self._solve_fallback(q, v, contact_forces)

        solve_time = (time.time() - start_time) * 1000

        # Update statistics
        self.solve_count += 1
        self.total_solve_time += solve_time

        # Extract solution
        if result['status'] == 'solved':
            x = result['x']
            ddq = x[:n_ddq]
            forces = x[n_ddq:n_ddq+n_forces]
            tau = x[n_ddq+n_forces:]

            # Map forces back to frames
            force_dict = {}
            for idx, contact_idx in enumerate(active_contacts):
                frame = self.contact_frames[contact_idx]
                force_dict[frame] = forces[3*idx:3*idx+3]
            for frame in self.contact_frames:
                if frame not in force_dict:
                    force_dict[frame] = np.zeros(3)
        else:
            # Fallback: gravity compensation
            ddq = np.zeros(n_ddq)
            tau = self._gravity_compensation(q)
            force_dict = {frame: np.zeros(3) for frame in self.contact_frames}

        return WBCResult(
            joint_torques=tau,
            joint_accelerations=ddq,
            contact_forces=force_dict,
            solve_time_ms=solve_time,
            status=result['status']
        )

    def _build_cost(
        self,
        q: np.ndarray,
        v: np.ndarray,
        n_ddq: int,
        n_forces: int,
        n_tau: int,
        active_contacts: List[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build QP cost matrices"""
        n_vars = n_ddq + n_forces + n_tau

        P = np.zeros((n_vars, n_vars))
        q_vec = np.zeros(n_vars)

        # Task costs
        for name, task in self.tasks.items():
            if not task.active or task.weight == 0.0:
                continue

            # Get task Jacobian and desired acceleration
            J = task.compute_jacobian(q, self.robot)
            a_des = task.compute_desired_acceleration(q, v, self.robot, task.gains)

            # Cost: w * ||J * ddq - a_des||^2
            # = w * (ddq' J' J ddq - 2 a_des' J ddq + a_des' a_des)
            w = task.weight

            # Quadratic term
            JtJ = J.T @ J
            P[:n_ddq, :n_ddq] += w * JtJ

            # Linear term
            q_vec[:n_ddq] -= w * J.T @ a_des

        # Force tracking (track MPC reference)
        force_weight = 0.01
        for idx, contact_idx in enumerate(active_contacts):
            frame = self.contact_frames[contact_idx]
            f_des = self.desired_forces.get(frame, np.zeros(3))

            f_start = n_ddq + 3 * idx
            for j in range(3):
                P[f_start+j, f_start+j] += force_weight
                q_vec[f_start+j] -= force_weight * f_des[j]

        # Regularization on torque
        torque_weight = 1e-4
        tau_start = n_ddq + n_forces
        for j in range(n_tau):
            P[tau_start+j, tau_start+j] += torque_weight

        # Make symmetric
        P = 0.5 * (P + P.T)

        return P, q_vec

    def _build_dynamics_constraint(
        self,
        q: np.ndarray,
        v: np.ndarray,
        n_ddq: int,
        n_forces: int,
        n_tau: int,
        active_contacts: List[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build dynamics equality constraint
        M * ddq + h = S' * tau + Jc' * f
        """
        n_vars = n_ddq + n_forces + n_tau
        n_contacts = len(active_contacts)

        # Get dynamics terms
        M = self.robot.get_mass_matrix()
        h = self.robot.get_nonlinear_effects()

        # Selection matrix (maps joint torques to generalized forces)
        # For floating base: S = [0_{6 x n_joints}; I_{n_joints}]
        S = np.zeros((self.nv, self.n_joints))
        S[6:, :] = np.eye(self.n_joints)

        # Contact Jacobian (stacked)
        Jc_list = []
        for contact_idx in active_contacts:
            frame = self.contact_frames[contact_idx]
            J = self.robot.get_frame_jacobian(frame)
            Jc_list.append(J[:3, :])  # Only linear part

        if Jc_list:
            Jc = np.vstack(Jc_list)
        else:
            Jc = np.zeros((0, self.nv))

        # Build constraint: M * ddq - S * tau - Jc' * f = -h
        A_eq = np.zeros((self.nv, n_vars))
        A_eq[:, :n_ddq] = M
        A_eq[:, n_ddq:n_ddq+n_forces] = -Jc.T if Jc.size > 0 else np.zeros((self.nv, n_forces))
        A_eq[:, n_ddq+n_forces:] = -S

        b_eq = -h

        return A_eq, b_eq

    def _build_inequality_constraints(
        self,
        q: np.ndarray,
        v: np.ndarray,
        n_ddq: int,
        n_forces: int,
        n_tau: int,
        active_contacts: List[int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build inequality constraints"""
        return self.constraints.build_constraint_matrices(
            q, v, self.robot,
            n_ddq=n_ddq,
            n_forces=n_forces,
            n_tau=n_tau,
            contact_indices=active_contacts
        )

    def _solve_qp(
        self,
        P: np.ndarray,
        q: np.ndarray,
        A: np.ndarray,
        l: np.ndarray,
        u: np.ndarray
    ) -> Dict:
        """Solve QP using OSQP"""
        P_sparse = sparse.csc_matrix(P)
        A_sparse = sparse.csc_matrix(A)

        if not self.solver_initialized:
            self.solver = osqp.OSQP()
            self.solver.setup(
                P=P_sparse,
                q=q,
                A=A_sparse,
                l=l,
                u=u,
                verbose=False,
                warm_start=True,
                eps_abs=1e-4,
                eps_rel=1e-4,
                max_iter=100
            )
            self.solver_initialized = True
        else:
            self.solver.update(
                Px=P_sparse.data,
                q=q,
                Ax=A_sparse.data,
                l=l,
                u=u
            )

        result = self.solver.solve()

        if result.info.status == 'solved':
            return {'x': result.x, 'status': 'solved'}
        else:
            return {'x': None, 'status': 'failed'}

    def _solve_fallback(
        self,
        q: np.ndarray,
        v: np.ndarray,
        contact_forces: Optional[Dict[str, np.ndarray]]
    ) -> Dict:
        """Fallback solution when QP solver unavailable"""
        tau = self._gravity_compensation(q)
        x = np.concatenate([np.zeros(self.nv), np.zeros(6), tau])
        return {'x': x, 'status': 'fallback'}

    def _gravity_compensation(self, q: np.ndarray) -> np.ndarray:
        """Compute gravity compensation torques"""
        g = self.robot.get_gravity_vector()
        # Extract joint torques (skip floating base)
        return g[6:6+self.n_joints]

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
        """Reset controller state"""
        self.solve_count = 0
        self.total_solve_time = 0.0
        self.solver_initialized = False
        self.solver = None
