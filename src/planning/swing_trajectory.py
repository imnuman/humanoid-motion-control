#!/usr/bin/env python3
"""
Swing Leg Trajectory Generation
Smooth trajectories for foot motion during swing phase

Author: Al Numan
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from abc import ABC, abstractmethod


@dataclass
class SwingParams:
    """Parameters for swing trajectory"""
    start_pos: np.ndarray       # Start position (3D)
    end_pos: np.ndarray         # End position (3D)
    apex_height: float = 0.08   # Maximum height above ground
    duration: float = 0.3       # Swing duration (s)

    # Optional orientation targets
    start_orientation: Optional[np.ndarray] = None  # [roll, pitch, yaw]
    end_orientation: Optional[np.ndarray] = None


class SwingTrajectory(ABC):
    """Abstract base class for swing trajectories"""

    @abstractmethod
    def get_position(self, t: float) -> np.ndarray:
        """Get position at time t"""
        pass

    @abstractmethod
    def get_velocity(self, t: float) -> np.ndarray:
        """Get velocity at time t"""
        pass

    @abstractmethod
    def get_acceleration(self, t: float) -> np.ndarray:
        """Get acceleration at time t"""
        pass


class CubicSwingTrajectory(SwingTrajectory):
    """
    Cubic spline swing trajectory

    Uses cubic polynomials for horizontal motion
    and sinusoidal profile for vertical motion.
    """

    def __init__(self, params: SwingParams):
        """
        Initialize cubic swing trajectory

        Args:
            params: Swing parameters
        """
        self.params = params
        self.start = params.start_pos.copy()
        self.end = params.end_pos.copy()
        self.apex_height = params.apex_height
        self.duration = params.duration

        # Compute horizontal coefficients (cubic: a0 + a1*t + a2*t^2 + a3*t^3)
        # Boundary conditions: p(0)=start, p(T)=end, v(0)=0, v(T)=0
        T = self.duration

        # x and y: cubic with zero velocity at boundaries
        # p(t) = a0 + a1*t + a2*t^2 + a3*t^3
        # v(t) = a1 + 2*a2*t + 3*a3*t^2
        # p(0) = a0 = start
        # v(0) = a1 = 0
        # p(T) = a0 + a2*T^2 + a3*T^3 = end
        # v(T) = 2*a2*T + 3*a3*T^2 = 0

        # From v(T)=0: a2 = -3*a3*T/2
        # Substituting into p(T): a0 - 3/2*a3*T^3 + a3*T^3 = end
        # => a3 = 2*(end - start) / T^3
        # => a2 = -3*(end - start) / T^2

        self.coeffs_x = np.array([
            self.start[0],
            0.0,
            -3 * (self.start[0] - self.end[0]) / (T * T),
            2 * (self.start[0] - self.end[0]) / (T * T * T)
        ])

        self.coeffs_y = np.array([
            self.start[1],
            0.0,
            -3 * (self.start[1] - self.end[1]) / (T * T),
            2 * (self.start[1] - self.end[1]) / (T * T * T)
        ])

        # Ground height (average of start and end)
        self.ground_height = (self.start[2] + self.end[2]) / 2

    def get_position(self, t: float) -> np.ndarray:
        """Get foot position at time t"""
        t = np.clip(t, 0.0, self.duration)
        phase = t / self.duration

        # Horizontal: cubic polynomial
        x = (self.coeffs_x[0] + self.coeffs_x[1] * t +
             self.coeffs_x[2] * t * t + self.coeffs_x[3] * t * t * t)
        y = (self.coeffs_y[0] + self.coeffs_y[1] * t +
             self.coeffs_y[2] * t * t + self.coeffs_y[3] * t * t * t)

        # Vertical: sinusoidal profile
        # z = ground + apex_height * sin(pi * phase)
        z = self.ground_height + self.apex_height * np.sin(np.pi * phase)

        # Interpolate start/end heights
        z += (1 - phase) * (self.start[2] - self.ground_height)
        z += phase * (self.end[2] - self.ground_height)

        return np.array([x, y, z])

    def get_velocity(self, t: float) -> np.ndarray:
        """Get foot velocity at time t"""
        t = np.clip(t, 0.0, self.duration)
        T = self.duration
        phase = t / T

        # Horizontal velocity
        vx = (self.coeffs_x[1] + 2 * self.coeffs_x[2] * t +
              3 * self.coeffs_x[3] * t * t)
        vy = (self.coeffs_y[1] + 2 * self.coeffs_y[2] * t +
              3 * self.coeffs_y[3] * t * t)

        # Vertical velocity
        vz = self.apex_height * np.pi / T * np.cos(np.pi * phase)
        vz += (self.end[2] - self.start[2]) / T

        return np.array([vx, vy, vz])

    def get_acceleration(self, t: float) -> np.ndarray:
        """Get foot acceleration at time t"""
        t = np.clip(t, 0.0, self.duration)
        T = self.duration
        phase = t / T

        # Horizontal acceleration
        ax = 2 * self.coeffs_x[2] + 6 * self.coeffs_x[3] * t
        ay = 2 * self.coeffs_y[2] + 6 * self.coeffs_y[3] * t

        # Vertical acceleration
        az = -self.apex_height * (np.pi / T) ** 2 * np.sin(np.pi * phase)

        return np.array([ax, ay, az])


class BezierSwingTrajectory(SwingTrajectory):
    """
    Bezier curve swing trajectory

    Uses cubic Bezier curves for smooth motion with
    controllable intermediate waypoints.
    """

    def __init__(
        self,
        params: SwingParams,
        mid_height_ratio: float = 0.5
    ):
        """
        Initialize Bezier swing trajectory

        Args:
            params: Swing parameters
            mid_height_ratio: Where the apex occurs (0-1)
        """
        self.params = params
        self.start = params.start_pos.copy()
        self.end = params.end_pos.copy()
        self.apex_height = params.apex_height
        self.duration = params.duration
        self.mid_ratio = mid_height_ratio

        # Create control points for cubic Bezier
        # P0 = start, P3 = end
        # P1 and P2 are intermediate control points

        self.P0 = self.start.copy()
        self.P3 = self.end.copy()

        # Intermediate points
        mid_xy = 0.5 * (self.start[:2] + self.end[:2])
        ground_z = 0.5 * (self.start[2] + self.end[2])

        # P1: slightly forward from start, at apex height
        self.P1 = np.array([
            self.start[0] + 0.3 * (self.end[0] - self.start[0]),
            self.start[1] + 0.3 * (self.end[1] - self.start[1]),
            ground_z + self.apex_height
        ])

        # P2: slightly behind end, at apex height
        self.P2 = np.array([
            self.start[0] + 0.7 * (self.end[0] - self.start[0]),
            self.start[1] + 0.7 * (self.end[1] - self.start[1]),
            ground_z + self.apex_height
        ])

    def _bezier(self, t: float) -> np.ndarray:
        """Evaluate cubic Bezier curve at parameter t"""
        t = np.clip(t, 0.0, 1.0)
        t2 = t * t
        t3 = t2 * t
        mt = 1 - t
        mt2 = mt * mt
        mt3 = mt2 * mt

        return (mt3 * self.P0 +
                3 * mt2 * t * self.P1 +
                3 * mt * t2 * self.P2 +
                t3 * self.P3)

    def _bezier_derivative(self, t: float) -> np.ndarray:
        """Evaluate Bezier curve derivative"""
        t = np.clip(t, 0.0, 1.0)
        t2 = t * t
        mt = 1 - t
        mt2 = mt * mt

        return (3 * mt2 * (self.P1 - self.P0) +
                6 * mt * t * (self.P2 - self.P1) +
                3 * t2 * (self.P3 - self.P2))

    def _bezier_second_derivative(self, t: float) -> np.ndarray:
        """Evaluate Bezier curve second derivative"""
        t = np.clip(t, 0.0, 1.0)
        mt = 1 - t

        return (6 * mt * (self.P2 - 2 * self.P1 + self.P0) +
                6 * t * (self.P3 - 2 * self.P2 + self.P1))

    def get_position(self, t: float) -> np.ndarray:
        """Get foot position at time t"""
        phase = np.clip(t / self.duration, 0.0, 1.0)
        return self._bezier(phase)

    def get_velocity(self, t: float) -> np.ndarray:
        """Get foot velocity at time t"""
        phase = np.clip(t / self.duration, 0.0, 1.0)
        # Chain rule: dp/dt = dp/du * du/dt = dp/du * (1/T)
        return self._bezier_derivative(phase) / self.duration

    def get_acceleration(self, t: float) -> np.ndarray:
        """Get foot acceleration at time t"""
        phase = np.clip(t / self.duration, 0.0, 1.0)
        return self._bezier_second_derivative(phase) / (self.duration ** 2)


class SwingTrajectoryGenerator:
    """
    Factory for creating swing trajectories with different profiles
    """

    def __init__(
        self,
        trajectory_type: str = "bezier",
        default_apex_height: float = 0.08,
        default_duration: float = 0.3
    ):
        """
        Initialize trajectory generator

        Args:
            trajectory_type: "cubic" or "bezier"
            default_apex_height: Default swing height
            default_duration: Default swing duration
        """
        self.trajectory_type = trajectory_type
        self.default_apex_height = default_apex_height
        self.default_duration = default_duration

    def create_trajectory(
        self,
        start_pos: np.ndarray,
        end_pos: np.ndarray,
        apex_height: float = None,
        duration: float = None
    ) -> SwingTrajectory:
        """
        Create a swing trajectory

        Args:
            start_pos: Starting foot position
            end_pos: Target landing position
            apex_height: Maximum height (uses default if None)
            duration: Swing duration (uses default if None)

        Returns:
            SwingTrajectory object
        """
        params = SwingParams(
            start_pos=start_pos,
            end_pos=end_pos,
            apex_height=apex_height or self.default_apex_height,
            duration=duration or self.default_duration
        )

        if self.trajectory_type == "cubic":
            return CubicSwingTrajectory(params)
        elif self.trajectory_type == "bezier":
            return BezierSwingTrajectory(params)
        else:
            raise ValueError(f"Unknown trajectory type: {self.trajectory_type}")

    def get_trajectory_at_phase(
        self,
        trajectory: SwingTrajectory,
        phase: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get trajectory state at normalized phase [0, 1]

        Args:
            trajectory: Swing trajectory object
            phase: Phase in [0, 1]

        Returns:
            Tuple of (position, velocity, acceleration)
        """
        t = phase * trajectory.params.duration
        return (
            trajectory.get_position(t),
            trajectory.get_velocity(t),
            trajectory.get_acceleration(t)
        )


def demo_trajectories():
    """Demonstrate swing trajectory generation"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Create trajectories
    start = np.array([0.0, 0.1, 0.0])
    end = np.array([0.3, 0.1, 0.0])

    cubic_params = SwingParams(start, end, apex_height=0.08, duration=0.3)
    cubic_traj = CubicSwingTrajectory(cubic_params)

    bezier_params = SwingParams(start, end, apex_height=0.08, duration=0.3)
    bezier_traj = BezierSwingTrajectory(bezier_params)

    # Sample trajectories
    t_vals = np.linspace(0, 0.3, 100)

    cubic_pos = np.array([cubic_traj.get_position(t) for t in t_vals])
    bezier_pos = np.array([bezier_traj.get_position(t) for t in t_vals])

    # Plot
    fig = plt.figure(figsize=(12, 5))

    # 3D plot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(cubic_pos[:, 0], cubic_pos[:, 1], cubic_pos[:, 2], 'b-', label='Cubic')
    ax1.plot(bezier_pos[:, 0], bezier_pos[:, 1], bezier_pos[:, 2], 'r--', label='Bezier')
    ax1.scatter(*start, color='green', s=100, label='Start')
    ax1.scatter(*end, color='red', s=100, label='End')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()
    ax1.set_title('Swing Trajectories (3D)')

    # Height profile
    ax2 = fig.add_subplot(122)
    ax2.plot(t_vals, cubic_pos[:, 2], 'b-', label='Cubic')
    ax2.plot(t_vals, bezier_pos[:, 2], 'r--', label='Bezier')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Height (m)')
    ax2.legend()
    ax2.set_title('Height Profile')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('swing_trajectories.png', dpi=150)
    print("Saved swing_trajectories.png")


if __name__ == "__main__":
    demo_trajectories()
