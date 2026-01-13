"""
Simulation environments for humanoid control
MuJoCo-based physics simulation
"""

from .mujoco_env import MuJoCoEnv, EnvConfig

__all__ = ['MuJoCoEnv', 'EnvConfig']
