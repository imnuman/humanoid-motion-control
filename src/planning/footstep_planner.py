#!/usr/bin/env python3
"""
Footstep Planner for Bipedal Locomotion
Plans footstep locations based on velocity commands

Author: Al Numan
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum


class FootSide(Enum):
    LEFT = "left"
    RIGHT = "right"


@dataclass
class Footstep:
    """Single footstep representation"""
    side: FootSide
    position: np.ndarray      # [x, y, z] in world frame
    orientation: float        # Yaw angle
    timestamp: float          # When to land
    duration: float          # How long foot is on ground

    @property
    def pose(self) -> np.ndarray:
        """Get full pose [x, y, z, yaw]"""
        return np.array([
            self.position[0],
            self.position[1],
            self.position[2],
            self.orientation
        ])


@dataclass
class FootstepPlannerConfig:
    """Footstep planner configuration"""
    # Kinematic limits
    max_step_length: float = 0.4      # Maximum forward step
    max_step_width: float = 0.25      # Maximum lateral step
    max_step_yaw: float = 0.3         # Maximum yaw change per step

    # Nominal stance
    nominal_stance_width: float = 0.2
    nominal_step_length: float = 0.25

    # Timing
    default_stance_duration: float = 0.35
    default_swing_duration: float = 0.25

    # Safety margins
    min_step_length: float = 0.05
    collision_margin: float = 0.02


class FootstepPlanner:
    """
    Footstep planner using Raibert heuristic and kinematic limits

    Plans future footstep locations based on:
    - Current velocity command
    - Current robot state
    - Kinematic constraints
    """

    def __init__(self, config: FootstepPlannerConfig = None):
        """
        Initialize footstep planner

        Args:
            config: Planner configuration
        """
        self.config = config or FootstepPlannerConfig()

        # Current footstep plan
        self.footstep_plan: List[Footstep] = []

        # Current foot positions
        self.foot_positions = {
            FootSide.LEFT: np.array([0.0, self.config.nominal_stance_width/2, 0.0]),
            FootSide.RIGHT: np.array([0.0, -self.config.nominal_stance_width/2, 0.0])
        }

        # Next foot to swing
        self.next_swing_foot = FootSide.RIGHT

        # Current time
        self.current_time = 0.0

    def update(
        self,
        robot_position: np.ndarray,
        robot_yaw: float,
        velocity_cmd: np.ndarray,  # [vx, vy, yaw_rate]
        current_foot_positions: Dict[FootSide, np.ndarray],
        horizon: int = 4
    ) -> List[Footstep]:
        """
        Update footstep plan

        Args:
            robot_position: Current CoM position
            robot_yaw: Current robot yaw
            velocity_cmd: Velocity command [vx, vy, yaw_rate]
            current_foot_positions: Current foot positions
            horizon: Number of steps to plan

        Returns:
            List of planned footsteps
        """
        self.foot_positions = {k: v.copy() for k, v in current_foot_positions.items()}

        # Clear old plan
        self.footstep_plan = []

        # Plan footsteps
        swing_foot = self.next_swing_foot
        com_pos = robot_position[:2].copy()
        yaw = robot_yaw
        time = self.current_time

        for i in range(horizon):
            # Compute target footstep
            footstep = self._compute_footstep(
                swing_foot=swing_foot,
                com_position=com_pos,
                com_yaw=yaw,
                velocity_cmd=velocity_cmd,
                start_time=time
            )

            self.footstep_plan.append(footstep)

            # Update state for next step
            stance_foot = FootSide.RIGHT if swing_foot == FootSide.LEFT else FootSide.LEFT
            swing_foot = stance_foot  # Alternate feet

            # Propagate CoM
            step_duration = self.config.default_stance_duration + self.config.default_swing_duration
            com_pos += velocity_cmd[:2] * step_duration
            yaw += velocity_cmd[2] * step_duration
            time += step_duration

        return self.footstep_plan

    def _compute_footstep(
        self,
        swing_foot: FootSide,
        com_position: np.ndarray,  # [x, y]
        com_yaw: float,
        velocity_cmd: np.ndarray,
        start_time: float
    ) -> Footstep:
        """
        Compute single footstep using Raibert heuristic

        Args:
            swing_foot: Which foot is swinging
            com_position: Current/predicted CoM position
            com_yaw: Current/predicted yaw
            velocity_cmd: Velocity command
            start_time: When this step starts

        Returns:
            Computed footstep
        """
        # Rotation matrix for current yaw
        c, s = np.cos(com_yaw), np.sin(com_yaw)
        R = np.array([[c, -s], [s, c]])

        # Nominal hip offset (body frame)
        if swing_foot == FootSide.LEFT:
            hip_offset_body = np.array([0.0, self.config.nominal_stance_width/2])
        else:
            hip_offset_body = np.array([0.0, -self.config.nominal_stance_width/2])

        # Hip offset in world frame
        hip_offset_world = R @ hip_offset_body

        # Raibert heuristic: foot placement based on velocity
        # p_foot = p_com + hip_offset + k * v * T/2
        stance_time = self.config.default_stance_duration
        k_capture = 0.3  # Capture point gain

        vx, vy = velocity_cmd[:2]

        # Velocity-based offset
        velocity_offset = k_capture * np.array([vx, vy]) * stance_time

        # Yaw offset
        yaw_rate = velocity_cmd[2]
        step_yaw = com_yaw + yaw_rate * stance_time

        # Target position (2D)
        target_xy = com_position + hip_offset_world + velocity_offset

        # Apply kinematic limits
        target_xy = self._apply_limits(target_xy, com_position, com_yaw, swing_foot)

        # Create footstep
        return Footstep(
            side=swing_foot,
            position=np.array([target_xy[0], target_xy[1], 0.0]),
            orientation=step_yaw,
            timestamp=start_time + self.config.default_swing_duration,
            duration=self.config.default_stance_duration
        )

    def _apply_limits(
        self,
        target: np.ndarray,
        com_position: np.ndarray,
        com_yaw: float,
        swing_foot: FootSide
    ) -> np.ndarray:
        """Apply kinematic limits to footstep"""
        # Transform to body frame
        c, s = np.cos(-com_yaw), np.sin(-com_yaw)
        R_inv = np.array([[c, -s], [s, c]])

        relative_target = R_inv @ (target - com_position)

        # Apply limits in body frame
        # Forward/backward limit
        relative_target[0] = np.clip(
            relative_target[0],
            -self.config.max_step_length / 2,
            self.config.max_step_length
        )

        # Lateral limit (with stance width consideration)
        nominal_y = self.config.nominal_stance_width / 2
        if swing_foot == FootSide.LEFT:
            relative_target[1] = np.clip(
                relative_target[1],
                nominal_y - self.config.max_step_width / 2,
                nominal_y + self.config.max_step_width / 2
            )
            # Ensure minimum stance width
            relative_target[1] = max(relative_target[1], self.config.collision_margin)
        else:
            relative_target[1] = np.clip(
                relative_target[1],
                -nominal_y - self.config.max_step_width / 2,
                -nominal_y + self.config.max_step_width / 2
            )
            relative_target[1] = min(relative_target[1], -self.config.collision_margin)

        # Transform back to world frame
        c, s = np.cos(com_yaw), np.sin(com_yaw)
        R = np.array([[c, -s], [s, c]])

        return com_position + R @ relative_target

    def get_next_footstep(self) -> Optional[Footstep]:
        """Get the next planned footstep"""
        if self.footstep_plan:
            return self.footstep_plan[0]
        return None

    def get_footstep_at_time(self, time: float) -> Optional[Footstep]:
        """Get footstep active at given time"""
        for footstep in self.footstep_plan:
            if footstep.timestamp <= time < footstep.timestamp + footstep.duration:
                return footstep
        return None

    def pop_completed_footstep(self, current_time: float):
        """Remove footsteps that have completed"""
        while self.footstep_plan:
            footstep = self.footstep_plan[0]
            if current_time > footstep.timestamp + footstep.duration:
                self.footstep_plan.pop(0)
                # Update next swing foot
                self.next_swing_foot = (
                    FootSide.RIGHT if footstep.side == FootSide.LEFT
                    else FootSide.LEFT
                )
            else:
                break

    def set_next_swing_foot(self, foot: FootSide):
        """Set which foot will swing next"""
        self.next_swing_foot = foot

    def get_support_polygon(self) -> np.ndarray:
        """Get current support polygon vertices"""
        left = self.foot_positions[FootSide.LEFT][:2]
        right = self.foot_positions[FootSide.RIGHT][:2]

        # Simple line between feet for bipedal
        return np.array([left, right])

    def is_com_in_support(self, com_position: np.ndarray) -> bool:
        """Check if CoM is within support polygon"""
        left = self.foot_positions[FootSide.LEFT][:2]
        right = self.foot_positions[FootSide.RIGHT][:2]

        # For bipedal: check if CoM projection is between feet
        foot_vec = right - left
        com_vec = com_position[:2] - left

        # Project onto foot line
        t = np.dot(com_vec, foot_vec) / np.dot(foot_vec, foot_vec)

        # Check if projection is between feet (with margin)
        margin = 0.1
        return -margin < t < 1 + margin


def demo_footstep_planner():
    """Demonstrate footstep planning"""
    planner = FootstepPlanner()

    # Initial state
    robot_pos = np.array([0.0, 0.0, 1.0])
    robot_yaw = 0.0
    velocity_cmd = np.array([0.5, 0.0, 0.1])  # Forward + turning

    foot_positions = {
        FootSide.LEFT: np.array([0.0, 0.1, 0.0]),
        FootSide.RIGHT: np.array([0.0, -0.1, 0.0])
    }

    # Plan footsteps
    plan = planner.update(
        robot_position=robot_pos,
        robot_yaw=robot_yaw,
        velocity_cmd=velocity_cmd,
        current_foot_positions=foot_positions,
        horizon=6
    )

    print("Footstep Plan:")
    print("-" * 60)
    for i, step in enumerate(plan):
        print(f"Step {i+1}: {step.side.value}")
        print(f"  Position: [{step.position[0]:.3f}, {step.position[1]:.3f}, {step.position[2]:.3f}]")
        print(f"  Yaw: {np.degrees(step.orientation):.1f} deg")
        print(f"  Time: {step.timestamp:.3f} s")


if __name__ == "__main__":
    demo_footstep_planner()
