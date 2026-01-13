#!/usr/bin/env python3
"""
Convex MPC Formulation using OSQP
QP problem construction for centroidal dynamics optimization
"""

import numpy as np
from scipy import sparse
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
import time

try:
    import osqp
    OSQP_AVAILABLE = True
except ImportError:
    OSQP_AVAILABLE = False
    print("Warning: OSQP not installed. MPC solver will be unavailable.")


@dataclass
class QPMatrices:
    """QP problem matrices: min 0.5 x'Px + q'x s.t. l <= Ax <= u"""
    P: sparse.csc_matrix  # Quadratic cost
    q: np.ndarray         # Linear cost
    A: sparse.csc_matrix  # Constraint matrix
    l: np.ndarray         # Lower bounds
    u: np.ndarray         # Upper bounds


@dataclass
class MPCWeights:
    """Cost function weights"""
    # State tracking weights
    position: np.ndarray = None      # (3,) [x, y, z]
    velocity: np.ndarray = None      # (3,) [vx, vy, vz]
    orientation: np.ndarray = None   # (3,) [roll, pitch, yaw]
    angular_velocity: np.ndarray = None  # (3,) [wx, wy, wz]

    # Control weights
    force: float = 1e-5              # Penalty on force magnitude
    force_rate: float = 1e-4         # Penalty on force rate of change

    def __post_init__(self):
        if self.position is None:
            self.position = np.array([10.0, 10.0, 100.0])
        if self.velocity is None:
            self.velocity = np.array([2.0, 2.0, 10.0])
        if self.orientation is None:
            self.orientation = np.array([50.0, 50.0, 10.0])
        if self.angular_velocity is None:
            self.angular_velocity = np.array([1.0, 1.0, 5.0])


class ConvexMPCFormulation:
    """
    Convex MPC problem formulation for humanoid locomotion

    Optimizes contact forces to track desired centroidal motion.
    Uses linearized centroidal dynamics and friction cone constraints.

    State: x = [p, theta, p_dot, omega] ∈ R^12
    Control: u = [f_1, f_2, ...] ∈ R^(3*n_contacts)
    """

    def __init__(
        self,
        mass: float = 47.0,
        gravity: float = 9.81,
        inertia: np.ndarray = None,
        horizon: int = 10,
        dt: float = 0.02,
        weights: MPCWeights = None
    ):
        """
        Initialize MPC formulation

        Args:
            mass: Robot mass
            gravity: Gravitational acceleration
            inertia: 3x3 rotational inertia matrix
            horizon: Prediction horizon
            dt: Timestep
            weights: Cost function weights
        """
        self.mass = mass
        self.gravity = gravity
        self.dt = dt
        self.horizon = horizon

        # Inertia matrix (approximation for humanoid)
        if inertia is None:
            self.inertia = np.diag([5.0, 5.0, 1.0])
        else:
            self.inertia = inertia
        self.inertia_inv = np.linalg.inv(self.inertia)

        # Weights
        self.weights = weights or MPCWeights()

        # State and control dimensions
        self.nx = 12  # [pos(3), euler(3), vel(3), ang_vel(3)]
        self.n_contacts = 2  # bipedal
        self.nu = 3 * self.n_contacts  # force per contact

        # Friction coefficient
        self.mu = 0.6

        # Force limits
        self.f_min = 10.0     # Minimum normal force
        self.f_max = 500.0    # Maximum normal force

        # OSQP solver
        self.solver = None
        self.solver_initialized = False

        # Build state weight matrix
        self._build_weight_matrices()

    def _build_weight_matrices(self):
        """Build state and control weight matrices"""
        # State weight (diagonal)
        Q_diag = np.concatenate([
            self.weights.position,
            self.weights.orientation,
            self.weights.velocity,
            self.weights.angular_velocity
        ])
        self.Q = np.diag(Q_diag)

        # Control weight
        self.R = np.eye(self.nu) * self.weights.force

    def build_dynamics_matrices(
        self,
        contact_positions: List[np.ndarray],
        contact_states: List[bool],
        yaw: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build linearized discrete-time dynamics matrices

        Centroidal dynamics:
            m * p_ddot = sum(f_i) - m * g * e_z
            I * omega_dot = sum(r_i x f_i)

        Discretized:
            x[k+1] = A * x[k] + B * u[k] + g_vec

        Args:
            contact_positions: List of contact positions relative to CoM
            contact_states: List of booleans (True if in contact)
            yaw: Current yaw angle for rotation

        Returns:
            Tuple of (A, B) matrices
        """
        dt = self.dt
        m = self.mass

        # Rotation matrix for yaw
        c, s = np.cos(yaw), np.sin(yaw)
        R_yaw = np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])

        # World-frame inertia
        I_world = R_yaw @ self.inertia @ R_yaw.T
        I_world_inv = np.linalg.inv(I_world)

        # State transition matrix
        A = np.eye(self.nx)
        A[0:3, 6:9] = np.eye(3) * dt    # pos from vel
        A[3:6, 9:12] = np.eye(3) * dt   # euler from ang_vel

        # Control matrix
        B = np.zeros((self.nx, self.nu))

        for i, (r, in_contact) in enumerate(zip(contact_positions, contact_states)):
            if not in_contact:
                continue

            # Force affects linear acceleration: a = f/m
            B[6:9, 3*i:3*i+3] = np.eye(3) * dt / m

            # Torque from force: tau = r x f, omega_dot = I^-1 * tau
            r_skew = np.array([
                [0, -r[2], r[1]],
                [r[2], 0, -r[0]],
                [-r[1], r[0], 0]
            ])
            B[9:12, 3*i:3*i+3] = I_world_inv @ r_skew * dt

        return A, B

    def build_friction_cone_constraints(
        self,
        contact_states: List[bool]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build friction cone constraints

        Approximates friction cone with 4-sided pyramid:
        -mu * fz <= fx <= mu * fz
        -mu * fz <= fy <= mu * fz
        fmin <= fz <= fmax

        Args:
            contact_states: List of contact states

        Returns:
            Tuple of (C, l, u) for constraints l <= C*u <= u
        """
        n_active = sum(contact_states)
        if n_active == 0:
            return np.zeros((0, self.nu)), np.zeros(0), np.zeros(0)

        # Each contact: 6 constraints (fx bounds, fy bounds, fz bounds)
        n_constraints = 6 * n_active
        C = np.zeros((n_constraints, self.nu))
        l = np.zeros(n_constraints)
        u = np.zeros(n_constraints)

        row = 0
        for i, in_contact in enumerate(contact_states):
            if not in_contact:
                continue

            # fx + mu*fz >= 0 => fx >= -mu*fz
            C[row, 3*i] = 1.0
            C[row, 3*i+2] = self.mu
            l[row] = 0.0
            u[row] = np.inf
            row += 1

            # -fx + mu*fz >= 0 => fx <= mu*fz
            C[row, 3*i] = -1.0
            C[row, 3*i+2] = self.mu
            l[row] = 0.0
            u[row] = np.inf
            row += 1

            # fy + mu*fz >= 0
            C[row, 3*i+1] = 1.0
            C[row, 3*i+2] = self.mu
            l[row] = 0.0
            u[row] = np.inf
            row += 1

            # -fy + mu*fz >= 0
            C[row, 3*i+1] = -1.0
            C[row, 3*i+2] = self.mu
            l[row] = 0.0
            u[row] = np.inf
            row += 1

            # fz >= fmin
            C[row, 3*i+2] = 1.0
            l[row] = self.f_min
            u[row] = self.f_max
            row += 1

            # Additional slack for stability
            C[row, 3*i+2] = -1.0
            l[row] = -self.f_max
            u[row] = -self.f_min
            row += 1

        return C[:row], l[:row], u[:row]

    def formulate_qp(
        self,
        x0: np.ndarray,
        x_ref: List[np.ndarray],
        contact_positions: List[np.ndarray],
        contact_schedule: List[List[bool]]
    ) -> QPMatrices:
        """
        Formulate the full MPC QP problem

        Decision variables: z = [u_0, u_1, ..., u_{N-1}]

        Args:
            x0: Initial state (12,)
            x_ref: Reference trajectory [(12,) * horizon]
            contact_positions: Contact positions relative to CoM
            contact_schedule: Contact states per timestep

        Returns:
            QPMatrices for OSQP
        """
        H = self.horizon
        nu = self.nu
        nx = self.nx

        # Total decision variables: forces at each timestep
        n_vars = nu * H

        # Build stacked dynamics
        # x[k] = A^k * x0 + sum_{j=0}^{k-1} A^{k-1-j} * B_j * u_j + gravity terms

        # Compute cost matrices
        # Cost: sum_k (x[k] - x_ref[k])' Q (x[k] - x_ref[k]) + u[k]' R u[k]

        # P matrix (quadratic cost on u)
        P_data = []
        P_row = []
        P_col = []

        # Build propagation matrices
        # Store power matrices A^k
        A_powers = [np.eye(nx)]
        for k in range(H):
            contacts = contact_schedule[k]
            A_k, _ = self.build_dynamics_matrices(
                contact_positions, contacts, x0[5]  # yaw
            )
            A_powers.append(A_k @ A_powers[-1])

        # Precompute B matrices for each step
        B_matrices = []
        for k in range(H):
            contacts = contact_schedule[k]
            _, B_k = self.build_dynamics_matrices(
                contact_positions, contacts, x0[5]
            )
            B_matrices.append(B_k)

        # Cost: state tracking + control effort
        # x[k] depends on all u[0..k-1]

        # Build Hessian P and gradient q
        P = np.zeros((n_vars, n_vars))
        q = np.zeros(n_vars)

        for k in range(H):
            x_k_ref = x_ref[k]

            # x[k] = A^k * x0 + sum_{j<k} A^{k-1-j} * B_j * u_j + gravity_effect
            # Cost contribution: (x[k] - x_ref)' Q (x[k] - x_ref)

            # Effect of u_j on x[k] is: A^{k-1-j} * B_j
            for j in range(k):
                effect_j = A_powers[k-1-j] @ B_matrices[j]

                for i in range(k):
                    effect_i = A_powers[k-1-i] @ B_matrices[i]

                    # P[j,i] += effect_j' Q effect_i
                    P_ji = effect_j.T @ self.Q @ effect_i
                    P[j*nu:(j+1)*nu, i*nu:(i+1)*nu] += P_ji

                # q[j] += 2 * effect_j' Q (A^k x0 - x_ref)
                x_k_free = A_powers[k] @ x0
                x_k_free[8] -= self.gravity * (k+1) * self.dt  # gravity on vz
                q_j = effect_j.T @ self.Q @ (x_k_free - x_k_ref)
                q[j*nu:(j+1)*nu] += 2 * q_j

            # Control effort
            P[k*nu:(k+1)*nu, k*nu:(k+1)*nu] += self.R

        # Make P symmetric
        P = 0.5 * (P + P.T)

        # Build constraints
        # 1. Friction cone constraints at each timestep
        # 2. Zero force for swing contacts

        constraint_rows = []
        constraint_l = []
        constraint_u = []

        for k in range(H):
            contacts = contact_schedule[k]

            # Friction cone constraints
            C_k, l_k, u_k = self.build_friction_cone_constraints(contacts)

            if C_k.shape[0] > 0:
                # Expand to full variable space
                C_full = np.zeros((C_k.shape[0], n_vars))
                C_full[:, k*nu:(k+1)*nu] = C_k
                constraint_rows.append(C_full)
                constraint_l.append(l_k)
                constraint_u.append(u_k)

            # Zero force for swing feet
            for i, in_contact in enumerate(contacts):
                if not in_contact:
                    # fx, fy, fz = 0
                    for j in range(3):
                        row = np.zeros(n_vars)
                        row[k*nu + 3*i + j] = 1.0
                        constraint_rows.append(row.reshape(1, -1))
                        constraint_l.append(np.array([0.0]))
                        constraint_u.append(np.array([0.0]))

        # Stack constraints
        if constraint_rows:
            A_constr = np.vstack(constraint_rows)
            l_constr = np.concatenate(constraint_l)
            u_constr = np.concatenate(constraint_u)
        else:
            A_constr = np.zeros((1, n_vars))
            l_constr = np.array([-np.inf])
            u_constr = np.array([np.inf])

        return QPMatrices(
            P=sparse.csc_matrix(P),
            q=q,
            A=sparse.csc_matrix(A_constr),
            l=l_constr,
            u=u_constr
        )

    def solve(
        self,
        x0: np.ndarray,
        x_ref: List[np.ndarray],
        contact_positions: List[np.ndarray],
        contact_schedule: List[List[bool]],
        warm_start: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Solve MPC optimization problem

        Args:
            x0: Initial state
            x_ref: Reference trajectory
            contact_positions: Contact positions
            contact_schedule: Contact schedule
            warm_start: Previous solution for warm starting

        Returns:
            Dict with optimal forces and solve info
        """
        if not OSQP_AVAILABLE:
            return self._solve_fallback(x0, x_ref, contact_positions, contact_schedule)

        start_time = time.time()

        # Formulate QP
        qp = self.formulate_qp(x0, x_ref, contact_positions, contact_schedule)

        # Setup or update solver
        if not self.solver_initialized:
            self.solver = osqp.OSQP()
            self.solver.setup(
                P=qp.P,
                q=qp.q,
                A=qp.A,
                l=qp.l,
                u=qp.u,
                verbose=False,
                warm_start=True,
                eps_abs=1e-4,
                eps_rel=1e-4,
                max_iter=100
            )
            self.solver_initialized = True
        else:
            self.solver.update(
                Px=qp.P.data,
                q=qp.q,
                Ax=qp.A.data,
                l=qp.l,
                u=qp.u
            )

        # Warm start if available
        if warm_start is not None:
            self.solver.warm_start(x=warm_start)

        # Solve
        result = self.solver.solve()

        solve_time = (time.time() - start_time) * 1000

        if result.info.status != 'solved':
            # Return gravity compensation as fallback
            f_gravity = self.mass * self.gravity / 2
            forces = np.zeros(self.nu * self.horizon)
            for k in range(self.horizon):
                for i, in_contact in enumerate(contact_schedule[k]):
                    if in_contact:
                        forces[k*self.nu + 3*i + 2] = f_gravity

            return {
                'forces': forces[:self.nu],
                'full_solution': forces,
                'solve_time_ms': solve_time,
                'status': 'fallback'
            }

        # Extract first timestep forces
        forces = result.x[:self.nu]

        return {
            'forces': forces,
            'full_solution': result.x,
            'solve_time_ms': solve_time,
            'status': result.info.status,
            'iterations': result.info.iter
        }

    def _solve_fallback(
        self,
        x0: np.ndarray,
        x_ref: List[np.ndarray],
        contact_positions: List[np.ndarray],
        contact_schedule: List[List[bool]]
    ) -> Dict:
        """Fallback solver when OSQP is not available"""
        # Simple gravity compensation
        f_gravity = self.mass * self.gravity / 2
        forces = np.zeros(self.nu)

        for i, in_contact in enumerate(contact_schedule[0]):
            if in_contact:
                forces[3*i + 2] = f_gravity

        return {
            'forces': forces,
            'full_solution': None,
            'solve_time_ms': 0.0,
            'status': 'fallback'
        }
