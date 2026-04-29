import torch
from datetime import datetime
import os
import time
import argparse
import numpy as np

from utils.utils import(
    ReplayBuffer,
    parse_arguments_from_ini,
    read_file_if_modified
)

from algorithms.sac import SAC

import wandb
import os.path as op

from envs.prex_isaac_env import PrexIsaacEnv

# --- 1 - Initializations -------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--no-wandb", action="store_true",
                    help="Disable Weights & Biases logging")
args_main = parser.parse_args()

RUN_NAME = datetime.now().strftime("%Y%m%d_%H%M%S")
MODELS_DIR = os.path.join("models", RUN_NAME)
LOGS_DIR = os.path.join("logs", RUN_NAME)

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

file_config_path = "config.ini"
args = parse_arguments_from_ini(file_config_path)
last_mod_time = os.path.getmtime(file_config_path)

if not args_main.no_wandb:
    wandb.init(
        project="prex_ultrasonic-sac",
        config={
            "learning_rate": 0.001,
            "architecture": "fc",
            "dataset": "coppelia-prex",
            "epochs": 0
        }
    )

TOTAL_TIMESTEPS = 1_000_000

device = "cuda"

# --- 2 - Creating environment --------------------------------------------
print("[Training] I - Creating environment...")

env = PrexIsaacEnv(
    max_episode_length=args["max_steps"],
    max_linear_speed=args["max_linear_speed"],
    max_angular_speed=args["max_angular_speed"],
    radius_target=args["radius_target"],
    physics_dt=1.0 / 60.0,
    rendering_dt=1.0,
    verbose=args["verbose"],
    clipping_limit=args["clipping_limit"],
    max_speed_bonus=args["max_speed_bonus"],
    repeating_action=args["repeating_action"],
    device=device,
    arena_geometry=[(2.0, 2.0), 0.2, 0.5],
)

print("[Training] ... Environment created")

# --- 3 - Creating replay buffer ------------------------------------------
print("[Training] II - Creating replay buffer...")

batch_size = args["batch_size"]
state_dim = env.observation_space.shape
action_dim = env.action_space.shape
replay_buffer = ReplayBuffer(
    capacity=args["replay_buffer_size"],
    batch_size=batch_size,
    state_shape=state_dim,
    action_shape=action_dim,
    device=device,
    normalize_rewards=False
)

print("[Training] ... Replay buffer created")

# --- 4 - Creating agent --------------------------------------------------
print("[Training] III - Creating agent...")

agent = SAC(
    env_name="prex_ultrasonic_robot",
    state_dim=state_dim,
    action_dim=action_dim,
    replay_buffer=replay_buffer,
    device=device,
    actor_lr=args["actor_lr"],
    critic_lr=args["critic_lr"],
    tau=args["tau"],
    alpha=args["alpha"],
    gamma=args["gamma"],
    action_bounds=(
        (-args["max_linear_speed"], -args["max_angular_speed"]),
        (args["max_linear_speed"], args["max_angular_speed"]),
    ),
)

print("[Training] ... Agent created")

# --- 5 - Training --------------------------------------------------------
print("[Training] Starting training ...")

tot_episodes = 0
timesteps = 0
save_on_episodes = args["save_on_episode"]
running_avg_reward = 0
running_avg_steps = 0

eps_return = 0
once = True
collect_random_timesteps = args["collect_random_steps"]

obs, _ = env.reset()

for _ in range (TOTAL_TIMESTEPS):
    entropy = 0

    if timesteps < collect_random_timesteps:
        action = np.random.uniform(low=-1, high=1, size=2)
        action *= env.max_bounds

    else:
        action, entropy = agent.select_action(
            torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        )
        action = action[0]

    action = np.round(action, 4)
    linear_vel = action[0]
    angular_vel = action[1]

    next_obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    running_avg_reward = (running_avg_reward * (timesteps) + reward) / (timesteps + 1)

    print(f"eps = {tot_episodes}, step_count = {timesteps}, reward = {reward:.3f}, "
        f"runn_avg_reward = {running_avg_reward}, distance = {env.dist:.3f}")

    eps_return += reward
    replay_buffer.add(obs, next_obs, action, reward, terminated)

    if timesteps >= collect_random_timesteps:
        entropy = agent.train(timesteps, device)

        if not args_main.no_wandb:
            wandb.log({
                "actor_loss": agent.actor_loss_value,
                "q_loss": agent.q_loss_value,
                "entropy": entropy
            })

    obs = next_obs.copy()

    if tot_episodes > 0 and tot_episodes % save_on_episodes == 0 and once:
        agent.save("", MODELS_DIR, tot_episodes)
        once = False

    if not args_main.no_wandb:
        wandb.log({
            "linear_action": linear_vel,
            "angular_action": angular_vel,
            "linear_speed_robot": env.linear_speed,
            "angular_speed_robot": env.angular_speed,
            "distance_to_center": env.dist,
            "reward": reward,
            "running_average_reward": running_avg_reward,
            "alpha": agent.alpha,
            "theta": env.theta
        })

    if done:
        running_avg_steps = (running_avg_steps * (tot_episodes) + env.step_counter) / (tot_episodes + 1)

        if not args_main.no_wandb:
            wandb.log({
                "ep_return": eps_return,
                "step_count": env.step_counter,
                "average_tot_steps": running_avg_steps
            })
        
        obs, _ = env.reset()
        tot_episodes += 1
        eps_return = 0.0
        once = True

        if timesteps >= collect_random_timesteps:
            agent.alpha = max(agent.alpha * args["alpha_decay_rate"], args["min_alpha"])

    timesteps += 1 