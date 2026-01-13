# Humanoid Motion Control

Whole-body motion control and locomotion for humanoid robots using model predictive control and reinforcement learning.

## Overview

This project implements a complete motion control stack for bipedal humanoid robots, combining classical model-based control with learned policies for robust locomotion and manipulation.

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    Humanoid Motion Control Architecture                       │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                         HIGH-LEVEL PLANNER                               │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐  │ │
│  │  │   Task       │  │  Footstep    │  │    Path      │  │   Gait     │  │ │
│  │  │  Planner     │──▶│  Planner     │──▶│  Planner     │──▶│ Scheduler  │  │ │
│  │  │             │  │              │  │              │  │            │  │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  └────────────┘  │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                         MID-LEVEL CONTROLLER                             │ │
│  │  ┌───────────────────────────────────────────────────────────────────┐ │ │
│  │  │                    Model Predictive Control (MPC)                  │ │ │
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │ │ │
│  │  │  │   Centroidal│  │    ZMP      │  │    Contact Force        │  │ │ │
│  │  │  │   Dynamics  │  │  Tracking   │  │    Optimization         │  │ │ │
│  │  │  └─────────────┘  └─────────────┘  └─────────────────────────┘  │ │ │
│  │  └───────────────────────────────────────────────────────────────────┘ │ │
│  │                              │                                         │ │
│  │                              ▼                                         │ │
│  │  ┌───────────────────────────────────────────────────────────────────┐ │ │
│  │  │                    Whole-Body Controller (WBC)                     │ │ │
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │ │ │
│  │  │  │  Inverse    │  │   Task      │  │    Constraint           │  │ │ │
│  │  │  │  Dynamics   │  │  Hierarchy  │  │    Handling             │  │ │ │
│  │  │  └─────────────┘  └─────────────┘  └─────────────────────────┘  │ │ │
│  │  └───────────────────────────────────────────────────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                         LOW-LEVEL CONTROL                                │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐  │ │
│  │  │    Joint     │  │   Torque     │  │    Motor     │  │   State    │  │ │
│  │  │  PD Control  │  │  Filtering   │  │   Drivers    │  │ Estimation │  │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  └────────────┘  │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Features

- **Convex MPC**: Real-time centroidal dynamics optimization at 500Hz
- **Whole-Body Control**: Task-space inverse dynamics with constraints
- **RL Locomotion**: PPO-trained walking policy with sim-to-real transfer
- **Push Recovery**: Reactive balance control under perturbations
- **Stair Climbing**: Perception-aware footstep planning
- **Manipulation**: Dual-arm coordination with locomotion
- **Teleoperation**: VR-based full-body motion retargeting

## Supported Robots

| Robot | DOF | Status |
|-------|-----|--------|
| Unitree H1 | 19 | Tested |
| Fourier GR-1 | 32 | Tested |
| Boston Dynamics Atlas (Sim) | 28 | Simulation |
| Custom Humanoid | Configurable | Supported |

## Installation

```bash
# Install dependencies
sudo apt install libeigen3-dev libosqp-dev liburdfdom-dev

# Install Pinocchio (robot dynamics library)
conda install -c conda-forge pinocchio

# Clone repository
git clone https://github.com/imnuman/humanoid-motion-control.git
cd humanoid-motion-control

# Create environment
conda create -n humanoid python=3.9
conda activate humanoid

# Install Python dependencies
pip install -r requirements.txt

# Build C++ components
mkdir build && cd build
cmake .. && make -j8
```

## Quick Start

### MPC Simulation

```bash
# Run MPC controller in MuJoCo
python src/mpc_sim.py --robot=h1 --task=walk

# Run with visualization
python src/mpc_sim.py --robot=h1 --task=walk --render
```

### RL Training

```bash
# Train walking policy in Isaac Gym
python src/train_rl.py --robot=h1 --task=walk --num_envs=4096

# Train stair climbing
python src/train_rl.py --robot=h1 --task=stairs --num_envs=4096

# Train push recovery
python src/train_rl.py --robot=h1 --task=push_recovery --num_envs=4096
```

### ROS2 Deployment

```bash
# Launch robot driver
ros2 launch humanoid_control robot.launch.py robot:=h1

# Launch MPC controller
ros2 launch humanoid_control mpc.launch.py

# Send walking command
ros2 topic pub /cmd_vel geometry_msgs/Twist \
    "{linear: {x: 0.3, y: 0.0}, angular: {z: 0.1}}"
```

## Control Modules

### Centroidal MPC

```python
from src.mpc import CentroidalMPC

mpc = CentroidalMPC(
    robot_model='h1',
    horizon=10,
    dt=0.02
)

# Set desired velocity
mpc.set_command(vx=0.5, vy=0.0, yaw_rate=0.1)

# Compute optimal forces
contact_forces = mpc.solve(current_state)
```

### Whole-Body Controller

```python
from src.wbc import WholeBodyController

wbc = WholeBodyController(robot_model='h1')

# Add tasks
wbc.add_task('com_tracking', weight=100.0, target=com_desired)
wbc.add_task('foot_pose', weight=50.0, target=foot_poses)
wbc.add_task('torso_orientation', weight=30.0, target=orientation)

# Solve for joint commands
joint_torques = wbc.solve(current_state, contact_forces)
```

### State Estimator

```python
from src.estimation import StateEstimator

estimator = StateEstimator(
    robot_model='h1',
    use_imu=True,
    use_kinematics=True
)

# Update with sensor data
estimator.update(imu_data, joint_states, contact_states)

# Get estimated state
state = estimator.get_state()
print(f"Base position: {state.position}")
print(f"Base velocity: {state.velocity}")
```

## Configuration

```yaml
# config/h1_config.yaml
robot:
  name: "unitree_h1"
  total_mass: 47.0
  standing_height: 1.0

mpc:
  horizon: 10
  dt: 0.02
  max_force: 500.0
  friction_coefficient: 0.6

  weights:
    position: [10.0, 10.0, 100.0]
    velocity: [1.0, 1.0, 1.0]
    angular_velocity: [1.0, 1.0, 10.0]
    force_rate: 0.001

wbc:
  tasks:
    com_tracking:
      weight: 100.0
      gain_p: [100, 100, 100]
      gain_d: [20, 20, 20]

    swing_foot:
      weight: 50.0
      gain_p: [200, 200, 200]
      gain_d: [40, 40, 40]

    torso_orientation:
      weight: 30.0
      gain_p: [100, 100, 100]
      gain_d: [20, 20, 20]

  constraints:
    friction_cone: true
    torque_limits: true
    joint_limits: true

gait:
  walk:
    stance_duration: 0.35
    swing_duration: 0.25
    swing_height: 0.08
    step_length: 0.3
```

## Project Structure

```
humanoid-motion-control/
├── src/
│   ├── mpc/
│   │   ├── centroidal_mpc.py      # Centroidal dynamics MPC
│   │   ├── mpc_formulation.py     # QP problem formulation
│   │   └── gait_scheduler.py      # Gait timing
│   ├── wbc/
│   │   ├── whole_body_controller.py
│   │   ├── task.py                # Task definitions
│   │   └── constraints.py         # Constraint handling
│   ├── estimation/
│   │   ├── state_estimator.py     # EKF state estimation
│   │   └── contact_estimator.py   # Contact detection
│   ├── planning/
│   │   ├── footstep_planner.py    # Footstep planning
│   │   └── swing_trajectory.py    # Swing leg trajectory
│   ├── rl/
│   │   ├── train.py               # RL training
│   │   ├── policy.py              # Policy network
│   │   └── env.py                 # Isaac Gym environment
│   └── utils/
│       ├── robot_model.py         # Pinocchio wrapper
│       └── math_utils.py          # Rotation utilities
├── models/
│   ├── h1/                        # H1 URDF and meshes
│   └── gr1/                       # GR-1 URDF and meshes
├── config/
│   ├── h1_config.yaml
│   └── gr1_config.yaml
├── urdf/
│   └── h1.urdf.xacro
├── cpp/
│   ├── mpc_solver.cpp             # C++ MPC solver
│   └── wbc_solver.cpp             # C++ WBC solver
└── requirements.txt
```

## Performance

| Metric | Value |
|--------|-------|
| Walking Speed | 1.2 m/s |
| MPC Solve Time | < 1 ms |
| WBC Solve Time | < 0.5 ms |
| Push Recovery | 150 N lateral |
| Stair Height | 20 cm |
| Control Rate | 500 Hz |

## Research Papers

This implementation is based on:
- [MIT Cheetah 3: Dynamic Balance Control](https://ieeexplore.ieee.org/document/8594448)
- [Whole-Body MPC for Humanoids](https://arxiv.org/abs/2010.08196)
- [Learning Human-like Locomotion](https://arxiv.org/abs/2402.18294)
- [Centroidal Dynamics](https://arxiv.org/abs/1603.02044)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

```bibtex
@software{humanoid_motion_control,
  author = {Al Numan},
  title = {Humanoid Motion Control: MPC + WBC for Bipedal Robots},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/imnuman/humanoid-motion-control}
}
```
