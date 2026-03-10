#!/usr/bin/env python3
import torch
from datetime import datetime
import os
import time
import numpy as np


from utils.utils import (
    ReplayBuffer,
    parse_arguments_from_ini,
    read_file_if_modified,
)


from algorithms.sac import SAC

import wandb
import os.path as op


import rclpy
from envs.prex_ultrasonic_sensor import PrexWorld

rclpy.init()


# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="prex_ultrasonic-sac",
    # track hyperparameters and run metadata
    config={
        "learning_rate": 0.001,
        "architecture": "fc",
        "dataset": "coppelia-prex",
        "epochs": 0,
    },
)


file_config_path = op.join(__file__[: -len("train.py")], "config.ini")
args = parse_arguments_from_ini(file_config_path)
last_mod_time = os.path.getmtime(file_config_path)

device = "cpu"#"cuda"
env = PrexWorld(
    max_episode_length=args["max_steps"],
    max_linear_speed=args["max_linear_speed"],
    max_angular_speed=args["max_angular_speed"],
    topic_pub=args["topic_pub"],
    type_ros2_msg=args["type_ros2_msg"],
    dt=args["dt"],
    verbose=args["verbose"],
    time_factor=args["time_factor"],
    clipping_limit=args["clipping_limit"],
    radius_target=args["radius_target"],
    max_speed_bonus=args["max_speed_bonus"],
    repeating_action=args["repeating_action"],
    # square
    perimeter=(2, 2),
)

# create replay buffer
batch_size = args["batch_size"]
n_sensors = args["n_sensors"]
state_dim = env.state_space  # Shape of state input (4, 84, 84)
action_dim = env.action_space
replay_buffer = ReplayBuffer(
    args["replay_buffer_size"],
    batch_size,
    state_dim,
    action_dim,
    normalize_rewards=False,
    device=device,
)

# define an agent
agent = SAC(
    "prex_ultrasonic_robot",
    env.state_space,
    env.action_space,
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

tot_episodes = 0
timesteps = 0
probability_training = 1.0
save_on_episodes = args["save_on_episode"]
running_avg_reward = 0
running_avg_steps = 0

folder_name = os.path.join("models", f"{datetime.now().strftime('%Y%m%d_%H%M%S')}")
path = args["path"]

eps_return = 0
once = True
collect_random_timesteps = args["collect_random_steps"]

obs, _, _, _ = env.reset()


def update(dt):
    global obs
    global tot_episodes
    global timesteps
    global probability_training
    global running_avg_reward
    global running_avg_steps
    global eps_return
    global once
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """
    entropy = 0
    if timesteps < collect_random_timesteps:
        action = np.random.uniform(low=-1, high=1, size=2)
        action *= env.max_bounds

    else:
        action, entropy = agent.select_action(
            torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        )
        action = action[0]

    v_linear = action[0]
    v_angular = action[1]
    # round action
    action = np.round(action, 4)

    next_obs, reward, _, done = env.step(action)
    running_avg_reward = (running_avg_reward * (timesteps) + reward) / (timesteps + 1)
    print(
        f"eps = {tot_episodes} step_count = {timesteps}, reward={reward:.3f}, runn_avg_reward={running_avg_reward:.3f}, distance={env.dist:.3f}"
    )

    eps_return += reward
    replay_buffer.add(obs, next_obs, action, reward, done)

    if timesteps >= collect_random_timesteps:
        entropy = agent.train(timesteps, device)
        wandb.log(
            {
                "actor_loss": agent.actor_loss_value,
                "q_loss": agent.q_loss_value,
                "entropy": entropy,
            }
        )

    obs = next_obs.copy()

    if tot_episodes > 0 and tot_episodes % save_on_episodes == 0 and once:
        agent.save(path, folder_name, tot_episodes)
        once = False

    wandb.log(
        {
            "v_linear_action": v_linear,
            "v_angular_action": v_angular,
            "v_linear_robot": env.linear_speed,
            "v_angular_robot": env.angular_speed,
            "dist_to_center": env.dist,
            "reward": reward,
            "runn_avg_reward": running_avg_reward,
            "alpha": agent.alpha,
            "theta": env.theta,
        }
    )

    if done is True:
        running_avg_steps = (running_avg_steps * (tot_episodes) + env.step_counter) / (
            tot_episodes + 1
        )

        wandb.log(
            {
                "ep_return": eps_return,
                "step_count": env.step_counter,
                "avg_tot_steps": running_avg_steps,
            }
        )

        obs, _, _, done = env.reset()
        tot_episodes += 1
        eps_return = 0.0
        once = True

        if timesteps >= collect_random_timesteps:
            # decay alpha
            agent.alpha = max(agent.alpha * args["alpha_decay_rate"], args["min_alpha"])

    timesteps += 1


dt = args["dt"]
env.dt = dt
while True:
    last_mod_time, args, has_changed = read_file_if_modified(
        args, file_config_path, last_mod_time
    )
    if has_changed:
        if env.dt != args["dt"]:
            env.dt = args["dt"]
            print(f"new env dt {args['dt']}")

        if agent.replay_buffer.batch_size != args["batch_size"]:
            agent.replay_buffer.change_bacth_size(args["batch_size"])
            agent.replay_buffer.batch_size = args["batch_size"]
            print(f"new batch_size {args['batch_size']}")

        if agent.replay_buffer.capacity != args["replay_buffer_size"]:
            agent.replay_buffer.increase_capacity(args["replay_buffer_size"])
            agent.replay_buffer.capacity = args["replay_buffer_size"]
            print(f"new replay buffer capcacity {args['replay_buffer_size']}")

        optimizer = agent.actor_optimizer
        if agent.actor_lr != args["actor_lr"]:
            for param_group in optimizer.param_groups:
                param_group["lr"] = args["actor_lr"]
            agent.actor_lr = args["actor_lr"]
            print(f"new actor_lr {args['actor_lr']}")

        optimizer = agent.q_optimizer
        if agent.critic_lr != args["critic_lr"]:
            for param_group in optimizer.param_groups:
                param_group["lr"] = args["critic_lr"]
            agent.critic_lr = args["critic_lr"]
            print(f"new critic_lr {args['critic_lr']}")

        if agent.tau != args["tau"]:
            agent.tau = args["tau"]
            print(f"new tau {args['tau']}")

        if agent.alpha != args["alpha"]:
            agent.alpha = args["alpha"]
            print(f"new alpha {args['alpha']}")

        if collect_random_timesteps != args["collect_random_steps"]:
            collect_random_timesteps = args["collect_random_steps"]
            print(f"new collect_random_steps {args['collect_random_steps']}")

        if save_on_episodes != args["save_on_episode"]:
            save_on_episodes = args["save_on_episode"]
            print(f"new save_on_episode {args['save_on_episode']}")

        if env.clipping_limit != args["clipping_limit"]:
            env.clipping_limit = args["clipping_limit"]
            print(f"new clipping_limit {args['clipping_limit']}")

        if env.radius_target != args["radius_target"]:
            env.radius_target = args["radius_target"]
            print(f"new radius_target {args['radius_target']}")

        if agent.gamma != args["gamma"]:
            agent.gamma = args["gamma"]
            print(f"new gamma {args['gamma']}")

        if env.max_speed_bonus != args["max_speed_bonus"]:
            env.max_speed_bonus = args["max_speed_bonus"]
            print(f"new max_speed_bonus {args['max_speed_bonus']}")

        if env.repeating_action != args["repeating_action"]:
            env.repeating_action = args["repeating_action"]
            print(f"new repeating_action {args['repeating_action']}")

    update(dt)
