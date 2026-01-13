#!/usr/bin/env python3
"""
Advanced Gait Scheduler for Bipedal Locomotion
Manages contact timing and phase for various gait patterns
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


class GaitPattern(Enum):
    """Available gait patterns"""
    STAND = "stand"           # Both feet in contact
    WALK = "walk"             # Alternating single support
    TROT = "trot"             # Diagonal support pattern
    RUN = "run"               # Flight phase included
    BOUND = "bound"           # Both feet together
    CRAWL = "crawl"           # Slow, stable walk


@dataclass
class GaitParams:
    """Parameters defining a gait pattern"""
    name: str
    stance_duration: float    # Time foot is on ground (s)
    swing_duration: float     # Time foot is in air (s)
    phase_offset: float       # Phase offset between legs [0, 1]
    swing_height: float       # Maximum swing height (m)
    step_length: float        # Nominal step length (m)
    has_flight_phase: bool = False  # True for running gaits


# Predefined gait parameters
GAIT_LIBRARY = {
    GaitPattern.STAND: GaitParams(
        name="stand",
        stance_duration=1.0,
        swing_duration=0.0,
        phase_offset=0.0,
        swing_height=0.0,
        step_length=0.0,
        has_flight_phase=False
    ),
    GaitPattern.WALK: GaitParams(
        name="walk",
        stance_duration=0.35,
        swing_duration=0.25,
        phase_offset=0.5,
        swing_height=0.08,
        step_length=0.30,
        has_flight_phase=False
    ),
    GaitPattern.TROT: GaitParams(
        name="trot",
        stance_duration=0.25,
        swing_duration=0.25,
        phase_offset=0.5,
        swing_height=0.10,
        step_length=0.35,
        has_flight_phase=False
    ),
    GaitPattern.RUN: GaitParams(
        name="run",
        stance_duration=0.15,
        swing_duration=0.35,
        phase_offset=0.5,
        swing_height=0.12,
        step_length=0.50,
        has_flight_phase=True
    ),
    GaitPattern.CRAWL: GaitParams(
        name="crawl",
        stance_duration=0.5,
        swing_duration=0.3,
        phase_offset=0.5,
        swing_height=0.05,
        step_length=0.15,
        has_flight_phase=False
    )
}


@dataclass
class FootState:
    """State of a single foot"""
    name: str
    in_contact: bool
    phase: float              # Phase within current mode [0, 1]
    time_in_mode: float       # Time spent in current mode
    time_to_transition: float # Time until next mode change
    position: np.ndarray      # Current position
    target_position: np.ndarray  # Target landing position


class AdvancedGaitScheduler:
    """
    Advanced gait scheduler with multiple gait patterns and smooth transitions

    Features:
    - Multiple gait patterns (walk, trot, run, etc.)
    - Smooth gait transitions
    - Adaptive timing based on velocity
    - Terrain-aware step planning
    """

    def __init__(
        self,
        gait: GaitPattern = GaitPattern.WALK,
        dt: float = 0.002
    ):
        """
        Initialize gait scheduler

        Args:
            gait: Initial gait pattern
            dt: Control timestep
        """
        self.dt = dt
        self.gait_params = GAIT_LIBRARY[gait].copy()
        self.current_gait = gait

        # Gait phase [0, 1)
        self.phase = 0.0
        self.period = self.gait_params.stance_duration + self.gait_params.swing_duration

        # Foot states
        self.feet = {
            'left': FootState(
                name='left',
                in_contact=True,
                phase=0.0,
                time_in_mode=0.0,
                time_to_transition=self.gait_params.stance_duration,
                position=np.array([0.0, 0.1, 0.0]),
                target_position=np.array([0.0, 0.1, 0.0])
            ),
            'right': FootState(
                name='right',
                in_contact=True,
                phase=self.gait_params.phase_offset,
                time_in_mode=0.0,
                time_to_transition=self.gait_params.stance_duration,
                position=np.array([0.0, -0.1, 0.0]),
                target_position=np.array([0.0, -0.1, 0.0])
            )
        }

        # Gait transition
        self.transitioning = False
        self.target_gait = None
        self.transition_progress = 0.0

    @property
    def gait(self) -> GaitPattern:
        return self.current_gait

    def set_gait(self, gait: GaitPattern, immediate: bool = False):
        """
        Set target gait pattern

        Args:
            gait: Target gait pattern
            immediate: If True, switch immediately; otherwise blend
        """
        if gait == self.current_gait:
            return

        if immediate:
            self.current_gait = gait
            self.gait_params = GAIT_LIBRARY[gait]
            self.period = self.gait_params.stance_duration + self.gait_params.swing_duration
            self.transitioning = False
        else:
            self.target_gait = gait
            self.transitioning = True
            self.transition_progress = 0.0

    def update(self, dt: Optional[float] = None) -> None:
        """
        Update gait phase

        Args:
            dt: Time step (uses default if None)
        """
        if dt is None:
            dt = self.dt

        # Update phase
        if self.period > 0:
            self.phase = (self.phase + dt / self.period) % 1.0

        # Update foot states
        self._update_foot_states()

        # Handle gait transition
        if self.transitioning:
            self._update_transition(dt)

    def _update_foot_states(self):
        """Update contact state for each foot"""
        stance_ratio = self.gait_params.stance_duration / self.period if self.period > 0 else 1.0

        for foot_name, offset in [('left', 0.0), ('right', self.gait_params.phase_offset)]:
            foot = self.feet[foot_name]
            foot_phase = (self.phase + offset) % 1.0

            # Determine contact state
            was_in_contact = foot.in_contact
            foot.in_contact = foot_phase < stance_ratio

            # Update phase within current mode
            if foot.in_contact:
                foot.phase = foot_phase / stance_ratio
                foot.time_to_transition = (stance_ratio - foot_phase) * self.period
            else:
                swing_phase = (foot_phase - stance_ratio) / (1.0 - stance_ratio)
                foot.phase = swing_phase
                foot.time_to_transition = (1.0 - foot_phase) * self.period

            # Detect transitions
            if was_in_contact and not foot.in_contact:
                # Touchdown to liftoff
                foot.time_in_mode = 0.0
            elif not was_in_contact and foot.in_contact:
                # Liftoff to touchdown
                foot.time_in_mode = 0.0
            else:
                foot.time_in_mode += self.dt

    def _update_transition(self, dt: float):
        """Handle smooth gait transition"""
        transition_time = 0.5  # seconds to transition
        self.transition_progress += dt / transition_time

        if self.transition_progress >= 1.0:
            # Transition complete
            self.current_gait = self.target_gait
            self.gait_params = GAIT_LIBRARY[self.target_gait]
            self.period = self.gait_params.stance_duration + self.gait_params.swing_duration
            self.transitioning = False
            self.target_gait = None
            self.transition_progress = 0.0
        else:
            # Blend parameters
            target_params = GAIT_LIBRARY[self.target_gait]
            t = self.transition_progress

            self.gait_params.stance_duration = (
                (1 - t) * GAIT_LIBRARY[self.current_gait].stance_duration +
                t * target_params.stance_duration
            )
            self.gait_params.swing_duration = (
                (1 - t) * GAIT_LIBRARY[self.current_gait].swing_duration +
                t * target_params.swing_duration
            )
            self.gait_params.swing_height = (
                (1 - t) * GAIT_LIBRARY[self.current_gait].swing_height +
                t * target_params.swing_height
            )

            self.period = self.gait_params.stance_duration + self.gait_params.swing_duration

    def get_contact_schedule(
        self,
        horizon: int,
        dt: float = None
    ) -> List[Dict[str, bool]]:
        """
        Get contact schedule for MPC horizon

        Args:
            horizon: Number of timesteps
            dt: Timestep (uses gait dt if None)

        Returns:
            List of contact states for each timestep
        """
        if dt is None:
            dt = self.dt

        schedule = []
        phase = self.phase

        stance_ratio = self.gait_params.stance_duration / self.period if self.period > 0 else 1.0

        for _ in range(horizon):
            # Left foot
            left_phase = phase
            left_in_contact = left_phase < stance_ratio

            # Right foot (with phase offset)
            right_phase = (phase + self.gait_params.phase_offset) % 1.0
            right_in_contact = right_phase < stance_ratio

            schedule.append({
                'left': left_in_contact,
                'right': right_in_contact
            })

            # Advance phase
            if self.period > 0:
                phase = (phase + dt / self.period) % 1.0

        return schedule

    def get_swing_progress(self, foot: str) -> float:
        """
        Get swing phase progress for a foot [0, 1]

        Args:
            foot: 'left' or 'right'

        Returns:
            Swing progress (0 at liftoff, 1 at touchdown)
        """
        if self.feet[foot].in_contact:
            return 0.0
        return self.feet[foot].phase

    def get_stance_progress(self, foot: str) -> float:
        """
        Get stance phase progress for a foot [0, 1]

        Args:
            foot: 'left' or 'right'

        Returns:
            Stance progress (0 at touchdown, 1 at liftoff)
        """
        if not self.feet[foot].in_contact:
            return 0.0
        return self.feet[foot].phase

    def get_next_touchdown_time(self, foot: str) -> float:
        """
        Get time until next touchdown

        Args:
            foot: 'left' or 'right'

        Returns:
            Time in seconds until touchdown
        """
        f = self.feet[foot]
        if f.in_contact:
            # Already in stance, return time to next swing + swing duration
            return f.time_to_transition + self.gait_params.swing_duration
        else:
            # In swing, return time to touchdown
            return f.time_to_transition

    def compute_foothold(
        self,
        foot: str,
        com_position: np.ndarray,
        com_velocity: np.ndarray,
        yaw: float = 0.0
    ) -> np.ndarray:
        """
        Compute target foothold position using Raibert heuristic

        Args:
            foot: 'left' or 'right'
            com_position: Current CoM position
            com_velocity: Current CoM velocity
            yaw: Current body yaw

        Returns:
            Target foothold position in world frame
        """
        # Nominal hip offset
        if foot == 'left':
            hip_offset = np.array([0.0, 0.1, 0.0])
        else:
            hip_offset = np.array([0.0, -0.1, 0.0])

        # Rotate by yaw
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        hip_world = R @ hip_offset

        # Raibert heuristic: p_foot = p_hip + k * v + step_length/2 * v_hat
        k = 0.03  # Tunable gain
        stance_time = self.gait_params.stance_duration

        # Velocity-based offset
        v_xy = com_velocity[:2]
        v_mag = np.linalg.norm(v_xy)

        if v_mag > 0.01:
            v_hat = v_xy / v_mag
            # Capture point: place foot to counteract velocity
            velocity_offset = k * v_mag * stance_time * np.append(v_hat, 0)
        else:
            velocity_offset = np.zeros(3)

        # Target position
        target = com_position + hip_world + velocity_offset

        # Ground projection
        target[2] = 0.0

        return target

    def get_swing_trajectory_params(
        self,
        foot: str,
        start_pos: np.ndarray,
        end_pos: np.ndarray
    ) -> Dict:
        """
        Get parameters for swing trajectory generation

        Args:
            foot: Foot name
            start_pos: Current foot position
            end_pos: Target landing position

        Returns:
            Dict with trajectory parameters
        """
        f = self.feet[foot]

        return {
            'start': start_pos,
            'end': end_pos,
            'apex_height': self.gait_params.swing_height,
            'duration': self.gait_params.swing_duration,
            'progress': f.phase if not f.in_contact else 0.0
        }

    def is_stable_for_transition(self) -> bool:
        """Check if current state is stable for gait transition"""
        # Prefer to transition when both feet are in stance
        return self.feet['left'].in_contact and self.feet['right'].in_contact

    def get_state_dict(self) -> Dict:
        """Get current gait state as dictionary"""
        return {
            'gait': self.current_gait.value,
            'phase': self.phase,
            'period': self.period,
            'transitioning': self.transitioning,
            'feet': {
                name: {
                    'in_contact': f.in_contact,
                    'phase': f.phase,
                    'time_to_transition': f.time_to_transition
                }
                for name, f in self.feet.items()
            }
        }
