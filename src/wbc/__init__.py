"""
Whole-Body Control modules for humanoid robots
Task-space inverse dynamics with hierarchical task composition
"""

from .whole_body_controller import WholeBodyController, WBCConfig
from .task import Task, TaskType, CoMTask, FootPoseTask, OrientationTask, JointRegularizationTask
from .constraints import Constraints, FrictionConeConstraint, TorqueLimitConstraint, JointLimitConstraint

__all__ = [
    'WholeBodyController',
    'WBCConfig',
    'Task',
    'TaskType',
    'CoMTask',
    'FootPoseTask',
    'OrientationTask',
    'JointRegularizationTask',
    'Constraints',
    'FrictionConeConstraint',
    'TorqueLimitConstraint',
    'JointLimitConstraint'
]
