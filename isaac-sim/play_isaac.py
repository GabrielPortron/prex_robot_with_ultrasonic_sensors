import torch
from datetime import datetime
import os
import time
import argparse
import numpy as np
import shutil

from utils.utils import(
    ReplayBuffer,
    parse_arguments_from_ini,
    read_file_if_modified,
    euler_to_quaternion,
    create_video
)

from PIL import Image

from algorithms.sac import SAC

import wandb
import os.path as op

from envs.prex_isaac_env import PrexIsaacEnv

# --- 1 - Initializations -------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--no-wandb", action="store_true",
                    help="Disable Weights & Biases logging")
parser.add_argument("--cube", action="store_true",
                    help="Spawns a cube for the training")
args_main = parser.parse_args()

RUN_NAME = datetime.now().strftime("%Y%m%d_%H%M%S")
MODELS_DIR = os.path.join("models", RUN_NAME)
LOGS_DIR = os.path.join("logs", RUN_NAME)

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

file_config_path = "config.ini" #when debug, add isaac-sim/ at the beginning of the path
args = parse_arguments_from_ini(file_config_path)
last_mod_time = os.path.getmtime(file_config_path)

# Uncomment below when debugging
args_main.no_wandb = True 

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
print("[Playing] I - Creating environment...")

env = PrexIsaacEnv(
    max_episode_length=args["max_steps"],
    max_linear_speed=args["max_linear_speed"],
    max_angular_speed=args["max_angular_speed"],
    radius_target=args["radius_target"],
    physics_dt=1.0/60.0,
    rendering_dt=1.0/15.0,
    verbose=args["verbose"],
    cube=args_main.cube,
    sensors=False,
    clipping_limit=args["clipping_limit"],
    max_speed_bonus=args["max_speed_bonus"],
    repeating_action=args["repeating_action"],
    device=device,
    arena_geometry=[(2.0, 2.0), 0.2, 0.5],
)

print("[Playing] ... Environment created")

# --- 3 - Creating replay buffer ------------------------------------------
print("[Playing] II - Creating replay buffer...")

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

print("[Playing] ... Replay buffer created")

# --- 4 - Creating agent --------------------------------------------------
print("[Playing] III - Creating agent...")

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

print("[Playing] ... Agent created")

from isaacsim.sensors.camera import Camera

camera = Camera(
    prim_path="/World/camera",
    position=np.array([0.0, 0.0, 5.0]),
    orientation=np.array(euler_to_quaternion(-180.0, -90.0, 0.0, degrees=True)),
    frequency=15,
    resolution=(1024, 1024)
)

camera.initialize()
camera.add_motion_vectors_to_frame()

print("[Evaluating] ... camera set up")

output_path = "/home/g.portron/gitRepos/prex_robot_with_ultrasonic_sensors/isaac-sim/records"
os.makedirs(output_path, exist_ok=True)

images_path = "/home/g.portron/gitRepos/prex_robot_with_ultrasonic_sensors/isaac-sim/records/images/"
os.makedirs(images_path, exist_ok=True)

print("loading weights...")
# Model with sensors :
# agent.load_weights("/home/g.portron/gitRepos/prex_robot_with_ultrasonic_sensors/isaac-sim/models/20260512_111705/prex_ultrasonic_robot_policy_3600_weights.pth")

# Model without sensors :
agent.load_weights("/home/g.portron/gitRepos/prex_robot_with_ultrasonic_sensors/isaac-sim/models/20260518_155021/prex_ultrasonic_robot_policy_1500_weights.pth")
agent.set_to_eval_mode()

# --- 5 - Playing -------------------------------------------------------------
print("[Playing] Starting playing ...")

tot_episodes = 0
timesteps = 0
save_on_episodes = args["save_on_episode"]
running_avg_reward = 0
running_avg_steps = 0

for eps in range (10):
    obs, _ = env.reset()

    tot_episodes += 1
    eps_return = 0.0
    done =False
    step = 0
    while not done:
        action, entropy = agent.select_action(
            torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        )
        action = action[0]

        action = np.round(action, 4)
        linear_vel = action[0]
        angular_vel = action[1]

        next_obs, reward, terminated, truncated, _ = env.step(action, render=True)
        done = terminated or truncated
        running_avg_reward = (running_avg_reward * (timesteps) + reward) / (timesteps + 1)

        print(f"eps = {tot_episodes}, step_count = {timesteps}, reward = {reward:.3f}, "
                f"position = {env.position[:2]}, "
            f"lin_vel = {linear_vel}, ang_vel = {angular_vel}, distance = {env.dist:.3f}")

        eps_return += reward
        obs = next_obs.copy()

        if not args_main.no_wandb:
            wandb.log({
                "linear_action": linear_vel,
                "angular_action": angular_vel,
                "linear_speed_robot": env.linear_speed,
                "angular_speed_robot": env.angular_speed,
                "distance_to_center": env.dist,
                "reward": reward,
                "running_average_reward": running_avg_reward,
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

        camera.get_current_frame()
        rgb_image = camera.get_rgb()
        image = Image.fromarray(rgb_image)
        save_path = f"{images_path}/ep{eps}_rgb_image_{step}.png"
        image.save(save_path)

        step += 1
        timesteps += 1

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
video_name = "episode_" + timestamp + ".mp4"
video_path = os.path.join(output_path, video_name)
create_video(video_path, images_path, fps=15)

shutil.rmtree(images_path)

print(f"Command to use to get the video : scp g.portron@10.163.11.19:{video_path} Téléchargements/")

env.close()