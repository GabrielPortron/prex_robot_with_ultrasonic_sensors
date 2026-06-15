# PREX Robot with Ultrasonic Sensors - RL Navigation Task

A reinforcement learning project implementing Soft Actor-Critic (SAC) and Proximal Policy Optimization (PPO) for autonomous robot navigation using ultrasonic sensors in Isaac Sim.

## Table of Contents
- [Project Overview](#project-overview)
- [Environment Details](#environment-details)
- [Getting Started](#getting-started)
- [Preparing a Simulation](#preparing-a-simulation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Towards a Real Robot](#towards-a-real-robot)
- [Installing Isaac Sim and Isaac Lab](#installing-isaac-sim-and-isaac-lab)
- [Troubleshooting](#troubleshooting)

---

## Project Overview

This project trains an [**iRobot Create3**](https://iroboteducation.github.io/create3_docs/) differential-drive robot to autonomously navigate toward the center of an arena using simulated ultrasonic sensors. The robot is trained entirely in **Isaac Sim** (NVIDIA's physics simulator), with the option to use either a custom SAC implementation or Stable Baselines3's PPO.

Features include:

- **Simulation**: Isaac Sim environments via the Gymnasium interface
- **Two RL algorithms**: Custom SAC (off-policy) and SB3 PPO (on-policy)
- **Ultrasonic Sensors**: Up to 4 simulated sensors using PhysX raycasting (no GPU rendering needed)
- **Obstacle Support**: Configurable number of randomly-placed cube obstacles, repositioned each episode
- **Safety Controller**: Hard-coded obstacle avoidance layer that overrides the policy when a sensor reading is critically low
- **Experiment Tracking**: Weights & Biases (wandb) integration

---

## Environment Details

**Robot Configuration:**
- iRobot Create3, differential-drive (controlled by linear and angular velocity)
  - Can move forward/backward (linear velocity along body x-axis)
  - Can rotate in place (angular velocity around z-axis)
  - Cannot move sideways
- Up to 4 simulated ultrasonic sensors: front, back, left, right
- Sensor range: 0.30 m (minimum) to 6.0 m (maximum)
- Arena: Optional L1×L2 m square enclosure with static walls
- Goal: A configurable 2D point `(x, y)`

**State Space (Observations):**

The observation is a 14-dimensional vector assembled in this order:

| Index | Component | Bounds | Description |
|---|---|---|---|
| 0–3 | `d_front, d_back, d_left, d_right` |0.3 to 6.0 m | Ultrasonic sensor distances |
| 4–6 | `x, y, z` | ±inf | Robot world-frame position |
| 7 | `yaw` | ±π rad | Robot heading |
| 8 | `vx` | ±0.5 m/s | Forward linear speed |
| 9 | `wz` | ±0.5 rad/s | Yaw rate |
| 10 | `cos(yaw)` | ±1.0 | Heading vector x-component |
| 11 | `sin(yaw)` | ±1.0 | Heading vector y-component |
| 12 | `delta` | ±π rad | Angle between heading and goal direction |
| 13 | `controller_flag` | True/False | True if the safety controller fired this step |

> **Note on `x, y, z` bounds:** Bounds are set to `±inf` because the `State` class uses them only to define the Gymnasium `Box` space; the robot will never actually reach large values during training.

**Action Space:**
- 2D continuous: `[Vx, Wz]`
- Bounds: Vx ∈ [−0.5, 0.5] m/s, Wz ∈ [−0.5, 0.5] rad/s (set in `config.ini`)

**Episode Termination:**

| Condition | Type | Default reward |
|---|---|---|
| `dist ≤ radius_target` (0.2 m) and moving forward | `terminated` | +10.0 |
| Robot flips (`z > 0.40 m`) or leaves bounds | `terminated` | −0.5 |
| `step_counter ≥ max_steps` (200) | `truncated` | −0.5 |

The robot is spawned at a **random position and heading** at the start of each episode. Cube obstacles (if any) are also randomly repositioned each episode, always away from the robot start position and the goal.

**Safety Controller:**

A hard-coded layer runs inside every `step()` call, before the policy action reaches the robot. If a sensor detects an obstacle closer than 0.40 m in the direction the robot is moving, the linear velocity is set to 0 and the angular velocity is kept unchanged (so the robot can rotate away). When this fires, `self.needed_control = True` for that step, which triggers a small penalty in the reward function to discourage the robot from relying on the override rather than learning avoidance itself.

---

## Getting Started

### Prerequisites
- Python 3.10+
- Isaac Sim (installed and licensed) — see [Installing Isaac Sim and Isaac Lab](#installing-isaac-sim-and-isaac-lab)
- The `env_isaaclab` virtual environment

### Setup

0. **Create the Isaac Lab environment** (first time only):

   Follow the [Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) to create the `env_isaaclab` virtual environment. See [Installing Isaac Sim and Isaac Lab](#installing-isaac-sim-and-isaac-lab) for a summary.

1. **Source the Isaac Lab environment:**
   ```bash
   source /home/<user>/env_isaaclab/bin/activate
   ```

2. **Install Python dependencies (with uv):**
   ```bash
   uv pip install -r requirements.txt
   ```

3. **(Optional) Configure Weights & Biases:**
   ```bash
   uv run wandb login
   ```
   To disable W&B logging without modifying code, either pass `--no-wandb` at runtime or set:
   ```bash
   export WANDB_MODE=disabled
   ```

---

## Preparing a Simulation

Before running a simulation, several components can be tuned to match your experimental setup. This section walks through each one.

### The State

The state is a major component in reinforcement learning — it defines what information the agent has access to at each timestep. Modifying it is a two-step process.

**Step 1 — Initialisation:** In the `__init__` method of `PrexIsaacEnv`, the state is declared as follows:

```python
state = State(
    sensors={
        "front": 6.0, "back": 6.0,
        "left":  6.0, "right": 6.0,
    },
    position={
        "x": inf, "y": inf, "z": inf,
    },
    orientation={
        "roll": None, "pitch": None, "yaw": math.pi,
    },
    linear_speed={
        "vx": max_linear_speed, "vy": None, "vz": None,
    },
    angular_speed={
        "wx": None, "wy": None, "wz": max_angular_speed,
    },
    heading_vec=(1.0, 1.0, math.pi),
    controller=True
)
```

Each component can be enabled by setting its bound value (refer to the table in [Environment Details](#environment-details)), or disabled by setting it to `None`. Make sure to respect the expected type for each field — for example, `heading_vec` is a tuple of 3 elements, so to disable it entirely use `heading_vec=None`. For the controller flag, set it to `True` to include it or `False` to exclude it.

**Step 2 — Reading the state:** In the `read_state` method, the state is populated at each step:

```python
state.update_values(
    sensors=sensors,
    position=position,
    orientation=yaw,
    linear_speed=linear_speed,
    angular_speed=angular_speed,
    heading_vec=heading_vector,
    controller=needed_control
)
```

This is where you decide what data is actually written into the state vector at runtime. Make sure this matches what you declared in the initialisation — particularly for the sensors. All values must be passed as `np.array([...])`. Most are already formatted correctly, but take care when adding new components.

### The Training Area

Two training configurations are supported.

**With arena (`--arena`):** The robot is confined inside a physical L1×L2 m walled enclosure. Wall dimensions can be adjusted in `config.ini` before starting a simulation:
```ini
arena_geometry = [(L1, L2), wall_depth, wall_height]
```

**Without arena:** The robot spawns inside a virtual L1×L2 m region with no physical borders. The robot can wander outside this region during training — it only defines the spawn area. This size can be adjusted in `config.ini`:
```ini
borderless_perimeter = (L1, L2)
```

### Cube Obstacles

The number of cube obstacles is specified at launch time via the `--cube` argument (see [Usage](#usage)). Cubes are static physical objects — the robot cannot push them. They are randomly repositioned at the start of each episode, always away from the robot and the goal. The minimal distance between each obstacle can be configured in `config.ini`.

The only practical limit on the number of cubes is the available space: the placement algorithm loops until it finds a valid non-overlapping position for each cube, so placing too many in a small area can cause the program to hang. In a 5×5 m area, no more than around 10 cubes is recommended.

Cube size can be changed in `config.ini`:
```ini
cube_dimension = 0.3    # side length in metres (default)
```

### Sensors

By default, the four sensors are orthogonal (each facing perpendicular to the others), which corresponds to `lateral_sensors_angle=180`. Reducing this value rotates the left and right sensors toward the front — the agent then has more information about what is ahead but loses coverage on the sides. For example, `lateral_sensors_angle=60` places the two lateral sensors within a 60° cone centred on the front direction. This only has an effect when using all four sensors.

All sensor physics parameters can be tuned in `config.ini`:

```ini
sensor_config = (180.0, 15.0, 5)  # lateral_sensors_angle / cone_angle / nb_rays
```

---

## Usage

> **Important:** always launch scripts with `uv run` after sourcing `env_isaaclab`. Using `python` directly may fail to resolve Isaac Sim imports.

### Train with SAC (default)

```bash
uv run train_isaac.py --cube <N>
```

Add `--arena` to enclose the scene with L1×L2 m walls:

```bash
uv run train_isaac.py --cube 0 --arena
uv run train_isaac.py --cube 3 --arena
```

Disable W&B:

```bash
uv run train_isaac.py --cube 2 --arena --no-wandb
```

### Train with PPO

```bash
uv run train_isaac.py --cube 1 --arena --ppo
```

### Evaluate / Record a Trained SAC Model

```bash
uv run evaluate_isaac.py --nb_episodes 10 --cube 2 --model <run_name> --weight <checkpoint>
```

- `--nb_episodes` is the number of episodes to run
- `--model` is the run folder name inside `models/` (e.g. `20260323_113058`)
- `--weight` is the episode checkpoint number (e.g. `3600` for `prex_ultrasonic_robot_policy_3600_weights.pth`)

After all evaluation episodes, a video is compiled at `records/episode_<model_name>.mp4`. To copy it to your local machine:

```bash
scp <user>@<server_ip>:<video_path> ~/Downloads/
```

### Evaluate a Trained PPO Model

```bash
uv run evaluate_isaac.py --nb_episodes 10 --cube 1 --model <run_name> --weight 0 --ppo
```

> For PPO, `--weight` is ignored — the final model is always loaded from `models/<run_name>/ppo_prex_final.zip`, alongside `vec_normalize.pkl` for observation normalisation statistics.

---

## Project Structure

```
algorithms/
├── sac.py                        # Custom SAC implementation
├── model.py                      # Policy and Q-value networks
├── callbacks/
│   ├── wandb_callback.py         # W&B logging callback for PPO
│   └── video_callback.py         # Episode recording callback for PPO             
isaac-sim/
├── envs/
│   ├── prex_isaac_env.py             # Main Gymnasium environment + State class
│   └── isaacsim_elements/
│       ├── arena.py                  # Rectangular walled arena (FixedCuboid walls)
│       └── cube.py                   # Cube obstacle manager
├── robots/
│   ├── differential_robot.py         # DifferentialRobot base class + Create3Robot
│   └── sensors/
│       └── ultrasonic_sensors.py     # PhysX-based ultrasonic sensor simulation
├── utils/
│   └── utils.py                      # ReplayBuffer, config parser, geometry helpers
├── train_isaac.py                    # Training entry point (SAC or PPO)
├── evaluate_isaac.py                 # Evaluation and video recording entry point
├── config.ini                        # All hyperparameters and run settings
coppelia/
├── envs/
│   ├── prex_ultrasonic_sensor.py        
├── prex/
│   ├── create_csv_file.py
│   ├── read_csv_file.py
│   ├── take_measures_4sensors.py
│   ├── take_measures_online.py
│   ├── temp_err.py
├── raspberry_pi5_scripts/
├── scene_coppelia/
├── utils/
│   └── utils.py
├── train.py                    
├── evaluate.py                 
├── config.ini                        
└── evaluate.py  
```

---

## Configuration

Edit `config.ini` to adjust hyperparameters. All values can be **hot-reloaded during SAC training** without restarting — the file is polled for changes at the end of each episode.

### [ENV]

| Key | Default | Description |
|---|---|---|
| `max_linear_speed` | 0.5 | Action bound for linear velocity (m/s) |
| `max_angular_speed` | 0.5 | Action bound for angular velocity (rad/s) |
| `repeating_action` | 20 | Physics steps simulated per `env.step()` call |
| `target_point` | (0.0, 0.0) | Coordinates of the target (m) |
| `radius_target` | 0.2 | Goal acceptance radius (m) |
| `borderless_perimeter` | (5.0, 5.0) | Robot and cube spawn region when not using arena (m) |
| `cube_dimension` | 0.3 | Side length of each cube obstacle (m) |
| `distance_between_objects` | 0.2 | Security distance between the spawned objects (m) |
| `arena_geometry` | [(2.0, 2.0), 0.2, 0.5] | Arena inner dimensions, wall depth, wall height (m) |
| `sensor_config` | (180.0, 15.0, 5) | Lateral sensor angle (°), cone angle (°), rays per sensor |

### [ALGORITHM]

| Key | Default | Description |
|---|---|---|
| `replay_buffer_size` | 50,000 | SAC replay buffer capacity (transitions) |
| `batch_size` | 256 | SAC training mini-batch size |
| `actor_lr` | 0.0005 | Actor network learning rate |
| `critic_lr` | 0.0005 | Critic network learning rate |
| `gamma` | 0.99 | Discount factor |
| `tau` | 0.005 | Polyak averaging coefficient for target networks |
| `alpha` | 0.2 | Initial entropy temperature |
| `alpha_decay_rate` | 0.999 | Per-episode multiplicative decay for alpha |
| `min_alpha` | 0.05 | Minimum entropy temperature |

### [SIMULATION]

| Key | Default | Description |
|---|---|---|
| `physics_dt` | 1/60 | The timestep between each new physical iteration in IsaacSim |
| `rendering_dt` | 1.0 | The timestep between each render iteration in IsaacSim. When training, it is recommended to keep it equal or above 1.0 (no render) |
| `max_steps` | 200 | Max steps per episode before truncation |
| `collect_random_steps` | 1000 | Steps of random exploration before SAC training begins |
| `save_on_episode` | 300 | SAC checkpoint frequency (in episodes) |
| `total_simulation_timesteps` | 2,000,000 | Total environment steps for a training run |

> **Note on `repeating_action`:** With `physics_dt = 1/60 s` and `repeating_action = 20`, each environment step corresponds to roughly 333 ms of simulated time. The robot has significant momentum between decisions. Lowering this value makes control more responsive but slows down wall-clock training time.

---

## Towards a Real Robot

This project currently has **no ROS 2 implementation** — all training and evaluation runs entirely inside Isaac Sim. The long-term goal is to deploy the trained policy on a physical iRobot Create3 using ROS 2.

The planned pipeline is:
1. Train the policy in Isaac Sim (this project)
2. Export the policy weights (already saved as `.pth` files)
3. Write a ROS 2 node that subscribes to the real ultrasonic sensor topic and the odometry topic, assembles the same state vector used during training, runs a forward pass through the policy network, and publishes the resulting `[Vx, Wz]` command as a `geometry_msgs/Twist` message on `/cmd_vel`

The Create3 already communicates via ROS 2 out of the box, and the sensor/odometry topic names are already referenced in the legacy `config.ini` fields (`topic_sensor`, `topic_odom`, `topic_pub`). The main sim-to-real challenge will be the **domain gap**: the simulated sensors use idealised raycasting, while real ultrasonic sensors have noise, blind spots, and multi-path reflections.

---

## Installing Isaac Sim and Isaac Lab

### Isaac Sim

Isaac Sim is NVIDIA's physics simulation platform, required to run this project. It requires a compatible NVIDIA GPU and driver.

- **Download and installation guide:** https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_workstation.html
- **System requirements:** https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html

Isaac Sim can be installed either through the **Omniverse Launcher** (recommended for a first install) or via a **pip-based installation** inside a Python environment.

### Isaac Lab

Isaac Lab is the reinforcement learning framework built on top of Isaac Sim. It provides the Python environment interface (`gymnasium`-compatible) and the `env_isaaclab` virtual environment used to run this project.

- **Installation guide:** https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html
- **GitHub repository:** https://github.com/isaac-sim/IsaacLab

The key step after installing Isaac Sim is to run Isaac Lab's setup script, which creates the `env_isaaclab` virtual environment with all required dependencies:

```bash
# From the IsaacLab repository root:
./isaaclab.sh --install
```

Once created, activate it before running any script in this project:

```bash
source /home/<user>/env_isaaclab/bin/activate
```

---

## Troubleshooting

### Isaac Sim fails to launch / `isaacsim` import error
Make sure `env_isaaclab` is sourced before calling `uv run`. Isaac Sim's Python bindings are only available inside that environment:
```bash
source /home/<user>/env_isaaclab/bin/activate
uv run train_isaac.py ...
```

### `CUDA out of memory`
The SAC replay buffer is pre-allocated on GPU. If you run out of memory, reduce `replay_buffer_size` in `config.ini` or switch the buffer to CPU by changing `device="cuda"` to `device="cpu"` in the `ReplayBuffer` constructor call inside `train_isaac.py`.

### Training seems stuck / reward not improving
- Check `distance_to_center` in W&B — if it is not decreasing after ~300k steps, the reward signal may not be informative enough
- Make sure `--arena` is passed — without walls the robot can wander far from the origin with no termination signal for a long time
- Check that `collect_random_steps` is not set too high; if the buffer is not filled, SAC training never starts

### W&B login / connection issues
Pass `--no-wandb` to disable logging entirely:
```bash
uv run train_isaac.py --cube 0 --arena --no-wandb
```