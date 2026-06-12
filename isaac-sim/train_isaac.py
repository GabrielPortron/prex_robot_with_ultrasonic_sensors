import torch
from datetime import datetime
import os
import argparse
import numpy as np

from utils.utils import(
    ReplayBuffer,
    parse_arguments_from_ini,
)

from algorithms.sac import SAC

import wandb

from envs.prex_isaac_env import PrexIsaacEnv

# --- 1 - Initializations -------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--no-wandb", action="store_true",
                    help="Disable Weights & Biases logging")
parser.add_argument("--cube", type=int, required=True,
                    help="Enter the number of cube you want to spawn for the training")
parser.add_argument("--arena", action="store_true",
                    help="The training will take place in an arena")
parser.add_argument("--ppo", action="store_true",
                    help="Change the training algorithm from SAC to PPO")
args_main = parser.parse_args()

RUN_NAME = datetime.now().strftime("%Y%m%d_%H%M%S")
MODELS_DIR = os.path.join("models", RUN_NAME)
LOGS_DIR = os.path.join("logs", RUN_NAME)

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

file_config_path = "config.ini" #when debug, add isaac-sim/ at the beginning of the path
args = parse_arguments_from_ini(file_config_path)

# Uncomment below when debugging
# args_main.no_wandb = True 
# args_main.cube = 1
# args_main.arena = True  

device = "cuda"


### --- PPO --- ###
if args_main.ppo:

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (
        CheckpointCallback,
        CallbackList,
    )
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from algorithms.callbacks.wandb_callback import PPOWandbCallback
    from algorithms.callbacks.video_callback import VideoCallback

    PPO_CONFIG = dict(
    learning_rate=3e-5,
    n_steps=4096,  
    batch_size=64,
    n_epochs=5,      
    gamma=0.87,     
    gae_lambda=0.99,     
    clip_range=0.1,      
    ent_coef=0.001,     
    vf_coef=0.5,
    target_kl=0.27,      
    max_grad_norm=0.5,
    policy_kwargs=dict(net_arch=[256, 256, 128]),
    verbose=args["verbose"],
    tensorboard_log=LOGS_DIR,
    device="cpu"
    )

    # --- 2 - Creating environment --------------------------------------------
    print("[Training] I - Creating environment...")

    env = PrexIsaacEnv(
        max_episode_length=args["max_steps"],
        max_linear_speed=args["max_linear_speed"],
        max_angular_speed=args["max_angular_speed"],
        radius_target=args["radius_target"],
        physics_dt=1.0/60.0,
        rendering_dt=1.0,
        verbose=args["verbose"],
        ppo=True,
        cube=args_main.cube,
        borderless_perimeter=args["borderless_perimeter"],
        cube_dimension=args["cube_dimension"],
        arena=args_main.arena,
        arena_geometry=args["arena_geometry"],
        sensor_config=args["sensor_config"],
        repeating_action=args["repeating_action"],
        device=device,
    )

    monitored_env = Monitor(env, filename=os.path.join(LOGS_DIR, "monitor.csv"))

    vec_env = DummyVecEnv([lambda: monitored_env])

    env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        gamma=0.995
    )

    print("[Training] ... Environment created")
    
    # --- 3 - Setting the callbacks ----------------------------------------
    callbacks = []

    if not args_main.no_wandb:
        wandb.init(
            project="prex_ultrasonic-sac",
            name=RUN_NAME,
            config={**PPO_CONFIG, "total_timesteps": args["total_simulation_timesteps"]},
            sync_tensorboard=True,
            save_code=True,
        )
        callbacks.append(PPOWandbCallback())

    callbacks.append(
        CheckpointCallback(
            save_freq=100_000,
            save_path=MODELS_DIR,
            name_prefix="prex_ultrasonic_robot_policy",
            verbose=1
        )
    )

    callback = CallbackList(callbacks)

    output_path = "/home/g.portron/gitRepos/prex_robot_with_ultrasonic_sensors/isaac-sim/records"
    os.makedirs(output_path, exist_ok=True)

    images_path = "/home/g.portron/gitRepos/prex_robot_with_ultrasonic_sensors/isaac-sim/records/images/"

    callbacks.append(
        VideoCallback(
            output_path=output_path,
            images_path=images_path,
            n_steps=args["max_steps"],
            fps=15
        )
    )

    # --- 4 - Creating agent -----------------------------------------------
    model = PPO("MlpPolicy", env, **PPO_CONFIG)

    # --- 5 - Training -----------------------------------------------------
    print("--- [Training with PPO] ---")
    try:
        model.learn(
            total_timesteps=args["total_simulation_timesteps"],
            callback=callback,
            reset_num_timesteps=True,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n[train_ppo] Training interrupted — saving current model.")

    # --- 6 - Save final model ---------------------------------------------
    model_path = os.path.join(MODELS_DIR, "ppo_prex_final")
    env_path = os.path.join(MODELS_DIR, "vec_normalize.pkl")
    model.save(model_path)
    env.save(env_path)

    print(f"[train_ppo] Final model saved to {model_path}.zip")
    print(f"[train_ppo] Final normalisation stats saved to {env_path}.zip")


### --- SAC --- ###
else:
    # --- 2 - Creating environment --------------------------------------------
    print("[Training] I - Creating environment...")

    env = PrexIsaacEnv(
        max_episode_length=args["max_steps"],
        max_linear_speed=args["max_linear_speed"],
        max_angular_speed=args["max_angular_speed"],
        radius_target=args["radius_target"],
        physics_dt=1.0/60.0,
        rendering_dt=1.0,
        verbose=args["verbose"],
        ppo=False,
        cube=args_main.cube,
        borderless_perimeter=args["borderless_perimeter"],
        cube_dimension=args["cube_dimension"],
        arena=args_main.arena,
        arena_geometry=args["arena_geometry"],
        sensor_config=args["sensor_config"],
        repeating_action=args["repeating_action"],
        device=device,
    )

    print("[Training] ... Environment created")

    # --- 3 - Creating replay buffer ---------------------------------------
    print("[Training] II - Creating replay buffer...")

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

    # --- 4 - Creating agent -----------------------------------------------
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

    # --- 5 - Training -----------------------------------------------------
    print("--- [Training with SAC] ---")
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

    for _ in range (args["total_simulation_timesteps"]):
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
            f"position = {env.position[:2]}, "
            f"lin_vel = {linear_vel}, ang_vel = {angular_vel}, distance = {env.dist:.3f}")

        eps_return += reward
        replay_buffer.add(
            obs[np.newaxis],
            next_obs[np.newaxis],
            action[np.newaxis],
            np.array([reward]),
            np.array([terminated]),
        )

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

if not args_main.no_wandb:
        wandb.finish()

env.close()