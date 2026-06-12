# PREX Robot with Ultrasonic Sensors - RL Navigation Task

A reinforcement learning project implementing Soft Actor-Critic (SAC) for autonomous robot navigation using 4 ultrasonic sensors in CoppeliaSim and real robot hardware.

## Table of Contents
- [Project Overview](#project-overview)
- [Student Assignment](#student-assignment)
- [Environment Details](#environment-details)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

## Project Overview

This project combines ROS 2 communication with Reinforcement Learning (SAC algorithm) to train a PrEx differential-drive robot for autonomous navigation using 4 ultrasonic sensors. The robot is simulated in CoppeliaSim and can be deployed on real hardware (Raspberry Pi 5).

Features include:

- **Simulation**: CoppeliaSim environments for training
- **Real Robot Support**: Integration with ROS 2 for real hardware control
- **Ultrasonic Sensors**: 4-sensor orthogonal array for distance sensing
- **Experiment Tracking**: Weights & Biases (wandb) integration

## Student Assignment

Your task is to complete the implementation of the reinforcement learning environment for robot navigation. The robot must learn to navigate towards the **center of a square arena** using **only the 4 ultrasonic sensor readings** (no obstacles are present, only the arena walls).

### **What You Need to Implement**

You need to complete five TODOs in `envs/prex_ultrasonic_sensor.py`. Specifically, replace the `...` placeholders with meaningful expressions or values to:

- define the state space,
- select and assign the state variables,
- implement the reward function,
- specify the success condition, and
- specify the failure condition.

Ensure each replacement matches the robot's differential-drive kinematics and the task objective (reach the arena center).

---

#### **TODO 1: Define the State Space**
```python
#TODO define the state space
self.state_space = ...
```

**Task:** Define `self.state_space` as a list/tuple representing the dimensions of your state representation.

**Available sensor data from ROS2:**
- `self.node_ros2.state = [d1, d2, d3, d4, vx, vy, vz, wx, wy, wz, yaw]`
  - `d1, d2, d3, d4`: Distances from 4 ultrasonic sensors (front, back, left, right)
  - Other data: velocities and orientation


**Important consideration:**
- **Think about the robot kinematics:** This is a differential-wheeled robot. Which components of the velocity are actually relevant for control?
  - For a 2-wheeled differential drive robot, only certain velocity components are meaningful
  - Linear velocity: Can the robot move sideways?
  - Angular velocity: Can the robot not tilt or roll ?

#### **TODO 2: Define the State Variables**
```python
#TODO define the state and all the following variables as you think is best for the task
# Example: select components from a numpy array from ROS2 state
# ros_state = np.array([d1, d2, d3, d4, vx, vy, vz, wx, wy, wz, yaw])
# 
# selected = ros_state[[6,7]]  # [vz,wx]

self.state = ...
self.dist = ...
self.position = ...
self.linear_speed = ...
self.angular_speed = ...
```

#### **TODO 3: Define the Reward Function**
```python
# TODO define the reward
reward = ...

if self.step_counter < ... :
   if ... <= self.radius_target:
      done = True
      self.info["terminate"] = "it reached the goal"
      #TODO consider wether or not to reward the robot if it completed the task
      reward += ...
else:
   done = True
   self.info["terminate"] = "it reached max episode length"
   #TODO consider wether or not to penalize the robot if it did not complete the task
   reward = ...
```

**Task:** Design a reward function that encourages the robot to reach the center of the arena.

---

#### **TODO 4: Define Success Condition**
```python
if ... <= self.radius_target:
      done = True
      self.info["terminate"] = "it reached the goal"
      #TODO consider wether or not to reward the robot if it completed the task
      reward += ...
```

**Task:** When the robot reaches the goal (within `self.radius_target` = 0.2), think if it worth to reward the agent for the completion.

---

#### **TODO 5: Define Failure Condition**
```python
else:
   done = True
   self.info["terminate"] = "it reached max episode length"
   #TODO consider wether or not to penalize the robot if it did not complete the task
   reward = ...
```

**Task:** When the episode ends due to ... without reaching the goal,think if it worth to penalize the agent.

---

### **Tips for Success**

1. **Start simple** 
2. **Test your reward** 
3. **Balance exploration:** The SAC algorithm handles exploration automatically via entropy term
4. **Monitor training:** Watch for increasing episode rewards and decreasing episode lengths
5. **Iterate:** If the robot doesn't learn, adjust your reward function and state representation

### **Deliverables**

- Completed `envs/prex_ultrasonic_sensor.py` with all TODOs implemented
- Brief report explaining your design choices:
  - State representation rationale
  - Reward function design and reasoning
  - Training results and observations

Good luck! 🤖

---


## Environment Details

**Robot Configuration:**
- Differential drive robot (controlled by linear and angular velocity)
  - **Note:** A differential-wheeled robot has constrained motion:
    - Can move forward/backward (linear velocity in x-direction)
    - Can rotate in place (angular velocity around z-axis)
    - Cannot move sideways or tilt
- 4 ultrasonic sensors: front, back, left, right
- Detection range: 0.30m (too close) to 4.0m (max range)
- Arena: Square/rectangular perimeter with walls (no obstacles inside)
- Goal: Center of the arena

**State Space (Observations):**
- (To be defined from the students)

**Action Space:**
- 2D continuous: `[linear_velocity, angular_velocity]`
- Range: [-max_linear_speed, +max_linear_speed], [-max_angular_speed, +max_angular_speed]
- Default: linear ∈ [-1.0, 1.0], angular ∈ [-1.0, 1.0] m/s or rad/s

**Episode Termination:**
- Success: Robot reaches within `radius_target` (0.2) of goal
- Failure: (To be defined from the students)
- The robot position is randomly reset at the start of each episode

## Getting Started

### Prerequisites
- Python 3.8+
- ROS 2 (Humble or later) for real robot control
- CoppeliaSim for simulation (download: https://www.coppeliarobotics.com/downloads)

### Setup

1. **Create and activate a virtual environment (with uv):**
   ```bash
   uv venv prexrl
   source prexrl/bin/activate
   ```

2. **Install Python dependencies (with uv):**
   ```bash
   uv pip install -r requirements.txt
   ```

3. **(Optional) Configure Weights & Biases (wandb):**
   ```bash
   uv run wandb login
   ```
   If you don’t want logging, set:
   ```bash
   export WANDB_MODE=disabled
   ```

4. **(Optional) Install ROS 2 Python packages:**
   ```bash
   sudo apt-get install python3-rclpy python3-geometry-msgs
   sudo apt-get install ros-<ros_distro>-tf-transformations
   uv pip install transforms3d
   ```

5. **Source ROS 2 (real robot only):**
   ```bash
   source /opt/ros/<ros_distro>/setup.bash
   ```

6. **Start CoppeliaSim simulation:**
   - Open `scene_coppelia/prex_square.ttt` in CoppeliaSim
   - Run the simulation

## Usage

**Train the agent:**
```bash
uv run python train.py
```

**Evaluate trained model:**
```bash
uv run python evaluate.py
```

**Run trained policy (play):**
```bash
uv run python play.py
```


## Project Structure

```
.
├── algorithms/
│   ├── sac.py              # SAC algorithm implementation
│   ├── model.py            # Policy and Q-value networks
│   ├── model_deeper.py     # Deeper network variant
│   └── model_deeper_less_neurons.py
├── envs/
│   └── prex_ultrasonic_sensor.py
├── utils/
│   └── utils.py            # Utilities (replay buffer, arg parsing, etc.)
├── prex/                   # Raspberry Pi hardware scripts
├── raspberry_pi5_scripts/  # ROS 2 nodes for real robot
├── scene_coppelia/         # CoppeliaSim scene files (.ttt)
├── train.py                # Main training script
├── play.py                 # Evaluation/play script
├── evaluate.py             # Evaluation helper
├── config.ini              # Configuration parameters
└── requirements.txt        # Python dependencies
```

## Configuration

Edit `config.ini` to adjust hyperparameters.

## Troubleshooting

### ModuleNotFoundError: No module named 'tf_transformations'
```bash
sudo apt-get install ros-<ros_distro>-tf-transformations
```

### ModuleNotFoundError: No module named 'transforms3d'
```bash
uv pip install transforms3d
```

### rclpy Import Fails
Ensure ROS 2 is sourced in your shell:
```bash
source /opt/ros/<ros_distro>/setup.bash
```
