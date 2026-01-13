"""
Model Predictive Control modules for humanoid locomotion
"""

from .centroidal_mpc import (
    CentroidalMPC,
    MPCConfig,
    RobotParams,
    State,
    ContactState,
    GaitScheduler
)
from .mpc_formulation import ConvexMPCFormulation
from .gait_scheduler import GaitPattern, AdvancedGaitScheduler

__all__ = [
    'CentroidalMPC',
    'MPCConfig',
    'RobotParams',
    'State',
    'ContactState',
    'GaitScheduler',
    'ConvexMPCFormulation',
    'GaitPattern',
    'AdvancedGaitScheduler'
]
