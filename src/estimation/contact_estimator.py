#!/usr/bin/env python3
"""
Contact Estimator for Humanoid Robots
Detects ground contact using force sensors, kinematics, and probability

Author: Al Numan
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, List
from collections import deque


@dataclass
class ContactEstimatorConfig:
    """Contact estimator configuration"""
    # Force threshold for contact detection
    force_threshold: float = 50.0  # N

    # Position-based threshold
    height_threshold: float = 0.02  # m

    # Velocity threshold
    velocity_threshold: float = 0.1  # m/s

    # Probability thresholds
    contact_probability_threshold: float = 0.7
    no_contact_probability_threshold: float = 0.3

    # Filter parameters
    filter_window: int = 5

    # Contact frames
    contact_frames: List[str] = None

    def __post_init__(self):
        if self.contact_frames is None:
            self.contact_frames = ['left_foot', 'right_foot']


@dataclass
class ContactState:
    """Contact state for a single foot"""
    in_contact: bool = False
    probability: float = 0.0
    normal_force: float = 0.0
    contact_position: np.ndarray = None
    time_in_state: float = 0.0

    def __post_init__(self):
        if self.contact_position is None:
            self.contact_position = np.zeros(3)


class ContactEstimator:
    """
    Multi-modal contact estimator

    Fuses multiple sources:
    1. Force/torque sensors (if available)
    2. Foot height from kinematics
    3. Foot velocity
    4. Expected contact from gait scheduler

    Uses Bayesian filtering to smooth transitions.
    """

    def __init__(
        self,
        robot_model,
        config: ContactEstimatorConfig = None
    ):
        """
        Initialize contact estimator

        Args:
            robot_model: Robot model for kinematics
            config: Estimator configuration
        """
        self.robot = robot_model
        self.config = config or ContactEstimatorConfig()

        # Contact states
        self.contact_states = {
            frame: ContactState()
            for frame in self.config.contact_frames
        }

        # History buffers for filtering
        self.force_history = {
            frame: deque(maxlen=self.config.filter_window)
            for frame in self.config.contact_frames
        }
        self.height_history = {
            frame: deque(maxlen=self.config.filter_window)
            for frame in self.config.contact_frames
        }

        # Expected contact from gait
        self.expected_contact = {frame: False for frame in self.config.contact_frames}

    def update(
        self,
        q: np.ndarray,
        v: np.ndarray,
        force_readings: Optional[Dict[str, np.ndarray]] = None,
        expected_contact: Optional[Dict[str, bool]] = None,
        dt: float = 0.002
    ) -> Dict[str, ContactState]:
        """
        Update contact estimates

        Args:
            q: Joint positions
            v: Joint velocities
            force_readings: Force sensor readings (optional)
            expected_contact: Expected contact from gait scheduler
            dt: Time step

        Returns:
            Updated contact states
        """
        # Update robot model
        self.robot.update_state(q, v)

        # Update expected contact
        if expected_contact:
            self.expected_contact = expected_contact.copy()

        for frame in self.config.contact_frames:
            # Compute evidence from each source
            prob_force = self._probability_from_force(frame, force_readings)
            prob_height = self._probability_from_height(frame)
            prob_velocity = self._probability_from_velocity(frame)
            prob_expected = self._probability_from_expected(frame)

            # Bayesian fusion
            # P(contact | evidence) ∝ P(evidence | contact) * P(contact)
            # Using log-odds formulation

            # Weights for each source
            w_force = 0.4 if force_readings else 0.0
            w_height = 0.3
            w_velocity = 0.2
            w_expected = 0.1 if expected_contact else 0.0

            # Normalize weights
            w_total = w_force + w_height + w_velocity + w_expected
            if w_total > 0:
                w_force /= w_total
                w_height /= w_total
                w_velocity /= w_total
                w_expected /= w_total

            # Weighted probability
            prob = (
                w_force * prob_force +
                w_height * prob_height +
                w_velocity * prob_velocity +
                w_expected * prob_expected
            )

            # Apply hysteresis
            current_state = self.contact_states[frame]
            if current_state.in_contact:
                threshold = self.config.no_contact_probability_threshold
                new_contact = prob > threshold
            else:
                threshold = self.config.contact_probability_threshold
                new_contact = prob > threshold

            # Update state
            if new_contact != current_state.in_contact:
                current_state.time_in_state = 0.0
            else:
                current_state.time_in_state += dt

            current_state.in_contact = new_contact
            current_state.probability = prob

            # Get contact position
            if new_contact:
                T = self.robot.get_frame_pose(frame)
                current_state.contact_position = T[:3, 3]

            # Get normal force estimate
            if force_readings and frame in force_readings:
                current_state.normal_force = force_readings[frame][2]
            else:
                current_state.normal_force = self._estimate_normal_force(frame, new_contact)

            self.contact_states[frame] = current_state

        return self.contact_states.copy()

    def _probability_from_force(
        self,
        frame: str,
        force_readings: Optional[Dict[str, np.ndarray]]
    ) -> float:
        """Compute contact probability from force sensor"""
        if force_readings is None or frame not in force_readings:
            return 0.5  # No information

        fz = force_readings[frame][2]  # Normal force

        # Add to history
        self.force_history[frame].append(fz)

        # Filtered force
        if len(self.force_history[frame]) > 0:
            fz_filtered = np.mean(self.force_history[frame])
        else:
            fz_filtered = fz

        # Sigmoid probability based on force threshold
        x = (fz_filtered - self.config.force_threshold) / (0.3 * self.config.force_threshold)
        prob = 1.0 / (1.0 + np.exp(-x))

        return prob

    def _probability_from_height(self, frame: str) -> float:
        """Compute contact probability from foot height"""
        # Get foot pose
        T = self.robot.get_frame_pose(frame)
        height = T[2, 3]

        # Add to history
        self.height_history[frame].append(height)

        # Filtered height
        if len(self.height_history[frame]) > 0:
            height_filtered = np.mean(self.height_history[frame])
        else:
            height_filtered = height

        # Probability: lower height = higher contact probability
        x = (self.config.height_threshold - height_filtered) / (0.5 * self.config.height_threshold)
        prob = 1.0 / (1.0 + np.exp(-x))

        return prob

    def _probability_from_velocity(self, frame: str) -> float:
        """Compute contact probability from foot velocity"""
        vel = self.robot.get_frame_velocity(frame)
        vel_mag = np.linalg.norm(vel[:3])  # Linear velocity magnitude

        # Low velocity = likely in contact
        x = (self.config.velocity_threshold - vel_mag) / (0.5 * self.config.velocity_threshold)
        prob = 1.0 / (1.0 + np.exp(-x))

        return prob

    def _probability_from_expected(self, frame: str) -> float:
        """Get expected contact probability from gait scheduler"""
        return 0.9 if self.expected_contact.get(frame, False) else 0.1

    def _estimate_normal_force(self, frame: str, in_contact: bool) -> float:
        """Estimate normal force when sensor not available"""
        if not in_contact:
            return 0.0

        # Rough estimate: weight distributed among contacts
        n_contacts = sum(1 for s in self.contact_states.values() if s.in_contact)
        if n_contacts > 0:
            total_mass = self.robot.total_mass if hasattr(self.robot, 'total_mass') else 47.0
            return total_mass * 9.81 / n_contacts

        return 0.0

    def get_contact_dict(self) -> Dict[str, bool]:
        """Get simple contact state dictionary"""
        return {
            frame: state.in_contact
            for frame, state in self.contact_states.items()
        }

    def get_contact_positions(self) -> Dict[str, np.ndarray]:
        """Get positions of contacts in world frame"""
        return {
            frame: state.contact_position.copy()
            for frame, state in self.contact_states.items()
            if state.in_contact
        }

    def get_normal_forces(self) -> Dict[str, float]:
        """Get estimated normal forces"""
        return {
            frame: state.normal_force
            for frame, state in self.contact_states.items()
        }

    def reset(self):
        """Reset contact estimates"""
        for frame in self.config.contact_frames:
            self.contact_states[frame] = ContactState()
            self.force_history[frame].clear()
            self.height_history[frame].clear()
