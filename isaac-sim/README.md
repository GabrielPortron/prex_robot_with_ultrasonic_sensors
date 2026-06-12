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
- [Troubleshooting](#troubleshooting)

---

## Project Overview

This project trains an **iRobot Create3** differential-drive robot to autonomously navigate toward the center of an arena using simulated ultrasonic sensors. The robot is trained entirely in **Isaac Sim** (NVIDIA's physics simulator), with the option to use either a custom SAC implementation or Stable Baselines3's PPO.

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
- Arena: Optional 2×2 m square enclosure with static walls
- Goal: World origin (centre of the arena)

**State Space (Observations):**

The observation is a 14-dimensional vector assembled in this order:

| Index | Component | Bounds | Description |
|---|---|---|---|
| 0–3 | `d_front, d_back, d_left, d_right` | ±20.0 m | Ultrasonic sensor distances |
| 4–6 | `x, y, z` | ±2.0 m | Robot world-frame position |
| 7 | `yaw` | ±π rad | Robot heading |
| 8 | `vx` | ±0.5 m/s | Forward linear speed |
| 9 | `wz` | ±0.5 rad/s | Yaw rate |
| 10 | `cos(yaw)` | ±1.0 | Heading vector x-component |
| 11 | `sin(yaw)` | ±1.0 | Heading vector y-component |
| 12 | `delta` | ±100 rad | Angle between heading and goal direction |
| 13 | `controller_flag` | ±1 | 1 if the safety controller fired this step |

**Action Space:**
- 2D continuous: `[linear_velocity, angular_velocity]`
- Bounds: linear ∈ [−0.5, 0.5] m/s, angular ∈ [−0.5, 0.5] rad/s (set in `config.ini`)

**Episode Termination:**

| Condition | Type | Default reward |
|---|---|---|
| `dist ≤ radius_target` (0.2 m) and moving forward | `terminated` | +10.0 |
| Robot flips (`z > 0.40 m`) or leaves bounds | `terminated` | −0.5 |
| `step_counter ≥ max_steps` (200) | `truncated` | −0.5 |

The robot is spawned at a **random position and heading** inside the arena at the start of each episode. Cube obstacles (if any) are also randomly repositioned each episode, always away from the robot start position and the goal.

**Safety Controller:**

A hard-coded layer runs inside every `step()` call, before the policy action reaches the robot. If a sensor detects an obstacle closer than 0.40 m in the direction the robot is moving, the linear velocity is set to 0 and the angular velocity is kept unchanged (so the robot can rotate away). When this fires, `self.needed_control = True` for that step, which triggers a small penalty in the reward function to discourage the robot from relying on the override rather than learning avoidance itself.

---

## Getting Started

### Prerequisites
- Python 3.10+
- Isaac Sim (installed and licensed)
- The `env_isaaclab` virtual environment sourced in your shell

### Setup

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

The state is a major component in reinforcement learning — it defines what information the robot has access to at each timestep. Modifying it is a two-step process.

**Step 1 — Initialisation:** In the `__init__` method of `PrexIsaacEnv`, the state is declared as follows:

```python
self.state = State(
    sensors={
        "front": 20.0, "back": 20.0,
        "left":  20.0, "right": 20.0,
    },
    position={
        "x": 2.0, "y": 2.0, "z": 2.0,
    },
    orientation={
        "roll": None, "pitch": None, "yaw": math.pi,
    },
    linear_speed={
        "vx": self.max_linear_speed, "vy": None, "vz": None,
    },
    angular_speed={
        "wx": None, "wy": None, "wz": self.max_angular_speed,
    },
    heading_vec=(1.0, 1.0, 100.0),
    controller=1
)
```

Each component can be enabled by setting its bound value (refer to the table in [Environment Details](#environment-details)), or disabled by setting it to `None`. Make sure to respect the expected type for each field — for example, `heading_vec` is a tuple of 3 elements, so to disable it entirely use `heading_vec=(None, None, None)`.

**Step 2 — Reading the state:** In the `read_state` method, the state is populated at each step:

```python
self.state.update_state(
    sensors=full_sensors,
    position=self.position,
    orientation=np.array([self.theta]),
    linear_speed=np.array([self.linear_speed]),
    angular_speed=np.array([self.angular_speed]),
    heading_vec=np.concatenate([self.heading_vec, [self.delta]]),
    controller=np.array([int(self.needed_control)])
)
```

This is where you decide what data is actually written into the state vector at runtime. Make sure this matches what you declared in the initialisation — particularly for the sensors. All values must be passed as `np.array([...])`. Most are already formatted correctly, but take care when adding new components.

### The Training Area

Two training configurations are supported.

**With arena (`--arena`):** The robot is confined inside a physical 2×2 m walled enclosure. Wall dimensions can be adjusted in `train_isaac.py` when creating the environment:
```python
arena_geometry=[(2.0, 2.0), 0.2, 0.5]  # (perimeter), wall depth, wall height
```

**Without arena:** The robot spawns inside a virtual 5×5 m region with no physical borders. The robot can wander outside this region during training — it only defines the spawn area. This size can be adjusted in the `__init__` method of `PrexIsaacEnv`:
```python
self.perimeter = np.array([5.0, 5.0])
```

### Cube Obstacles

The number of cube obstacles is specified at launch time via the `--cube` argument (see [Usage](#usage)). Cubes are static physical objects — the robot cannot push them. They are randomly repositioned at the start of each episode, always away from the robot and the goal.

The only practical limit on the number of cubes is the available space: the placement algorithm loops until it finds a valid non-overlapping position for each cube, so placing too many in a small area can cause the program to hang. In a 5×5 m area, no more than around 10 cubes is recommended.

Cube size (default 0.3×0.3×0.3 m) can be changed in the `launch` method of `PrexIsaacEnv`:
```python
self.cube = Cube(
    world=self.world,
    nb_cube=self.nb_cube,
    scale=(0.3, 0.3, 0.3),
    perimeter=self.perimeter,
)
```

### Sensors

By default, the four sensors are orthogonal (each facing perpendicular to the others), which corresponds to `fov=180`. Reducing this value rotates the left and right sensors toward the front, effectively giving the robot a wider forward field of view at the cost of rear coverage. For example, `fov=60` places all three non-rear sensors within a 60° frontal cone. This setting can be changed in the `launch` method of `PrexIsaacEnv`:

```python
self.sensors = UltrasonicSensors(nb_sensors=self.state.nb_sensors, fov=180)
```

This only has an effect when using all four sensors.

The physical simulation of each sensor (cone angle and number of rays) can also be tuned. By default, each sensor casts 5 rays spread over a 15° cone. These values can be changed in the `_cone_raycast` method of `ultrasonic_sensors.py`. If you change them, make sure to apply the same values in `draw_rays` so the viewport visualisation stays consistent.

---

## Usage

> **Important:** always launch scripts with `uv run` after sourcing `env_isaaclab`. Using `python` directly may fail to resolve Isaac Sim imports.

### Train with SAC (default)

```bash
uv run train_isaac.py --cube <N>
```

Add `--arena` to enclose the scene with 2×2 m walls (recommended):

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
uv run play_isaac.py --cube 2 --model <run_name> --weight <checkpoint>
```

- `--model` is the run folder name inside `models/` (e.g. `20260323_113058`)
- `--weight` is the episode number in the checkpoint filename (e.g. `3600` for `prex_ultrasonic_robot_policy_3600_weights.pth`)

After 10 evaluation episodes a video is compiled at `records/episode_<model_name>.mp4`. To copy it to your local machine:

```bash
scp <user>@<server_ip>:<video_path> ~/Downloads/
```

### Evaluate a Trained PPO Model

```bash
uv run play_isaac.py --cube 1 --model <run_name> --weight 0 --ppo
```

> For PPO, `--weight` is ignored — the final model is always loaded from `models/<run_name>/ppo_prex_final.zip`, alongside `vec_normalize.pkl` for observation normalisation statistics.

---

## Project Structure

```
isaac-sim/
├── algorithms/
│   ├── sac.py                        # Custom SAC implementation
│   ├── model.py                      # Policy and Q-value networks (PyTorch)
│   └── callbacks/
│       ├── wandb_callback.py         # W&B logging callback for PPO
│       └── video_callback.py         # Episode recording callback for PPO
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
├── play_isaac.py                     # Evaluation and video recording entry point
├── config.ini                        # All hyperparameters and run settings
└── requirements.txt                  # Python dependencies
```

---

## Configuration

Edit `config.ini` to adjust hyperparameters. The `[DEFAULT]` section contains values that are fixed for the task. The `[MODIFIABLE]` section contains values intended to be tuned — they can also be **hot-reloaded during training** without restarting (the file is polled for changes each episode).

### [DEFAULT]

| Key | Default | Description |
|---|---|---|
| `max_steps` | 200 | Max steps per episode before truncation |
| `max_linear_speed` | 0.5 | Action bound for linear velocity (m/s) |
| `max_angular_speed` | 0.5 | Action bound for angular velocity (rad/s) |

### [MODIFIABLE]

| Key | Default | Description |
|---|---|---|
| `repeating_action` | 20 | Physics steps simulated per `env.step()` call |
| `radius_target` | 0.2 | Goal acceptance radius (m) |
| `replay_buffer_size` | 50,000 | SAC replay buffer capacity (transitions) |
| `batch_size` | 256 | SAC training mini-batch size |
| `actor_lr` | 0.0005 | Actor network learning rate |
| `critic_lr` | 0.0005 | Critic network learning rate |
| `gamma` | 0.99 | Discount factor |
| `tau` | 0.005 | Polyak averaging coefficient for target networks |
| `alpha` | 0.2 | Initial entropy temperature |
| `alpha_decay_rate` | 0.9999 | Per-episode multiplicative decay for alpha |
| `min_alpha` | 0.05 | Minimum entropy temperature |
| `collect_random_steps` | 1000 | Steps of random exploration before SAC training begins |
| `save_on_episode` | 300 | SAC checkpoint frequency (in episodes) |

> **Note on `repeating_action`:** With `physics_dt = 1/60 s` and `repeating_action = 20`, each environment step corresponds to roughly 333 ms of simulated time. The robot has significant momentum between decisions. Lowering this value makes control more responsive but slows down wall-clock training time.

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