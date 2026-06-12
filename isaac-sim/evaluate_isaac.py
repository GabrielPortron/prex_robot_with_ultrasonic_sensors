import torch
from datetime import datetime
import os
import argparse
import numpy as np
import shutil

from utils.utils import(
    ReplayBuffer,
    parse_arguments_from_ini,
    euler_to_quaternion,
    create_video
)

from PIL import Image

from algorithms.sac import SAC

from envs.prex_isaac_env import PrexIsaacEnv

# --- 1 - Initializations -------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--cube", type=int, required=True,
                    help="Enter the number of cube you want to spawn for the training")
parser.add_argument("--ppo", action="store_true",
                    help="Change the algorithm from SAC to PPO")
parser.add_argument("--arena", action="store_true",
                    help="The training will take place in 2.0x2.0 m2 arena")
parser.add_argument("--model",    type=str, required=True,
                    help="Path to the saved model zip (without .zip extension)")
parser.add_argument("--weight",    type=int, required=True,
                    help="'Age' of the model")
parser.add_argument("--nb_episodes",    type=int, required=True,
                    help="Number of episodes for the evaluation")
args_main = parser.parse_args()

RUN_NAME = datetime.now().strftime("%Y%m%d_%H%M%S")

file_config_path = "config.ini" #when debug, add isaac-sim/ at the beginning of the path
args = parse_arguments_from_ini(file_config_path)

device = "cuda"

model_path = "/home/g.portron/gitRepos/prex_robot_with_ultrasonic_sensors/isaac-sim/models/" + args_main.model + f"/prex_ultrasonic_robot_policy_{args_main.weight}_weights.pth"


### --- PPO --- ###
if args_main.ppo:

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    # --- 2 - Creating environment --------------------------------------------
    print("[Playing] I - Creating environment...")

    env = PrexIsaacEnv(
        max_episode_length=args["max_steps"],
        max_linear_speed=args["max_linear_speed"],
        max_angular_speed=args["max_angular_speed"],
        target_point=args["target_point"],
        radius_target=args["radius_target"],
        physics_dt=1.0/60.0,
        rendering_dt=1.0,
        verbose=args["verbose"],
        ppo=True,
        cube=args_main.cube,
        borderless_perimeter=args["borderless_perimeter"],
        cube_dimension=args["cube_dimension"],
        dist_objects=args["distance_between_objects"],
        arena=args_main.arena,
        arena_geometry=args["arena_geometry"],
        sensor_config=args["sensor_config"],
        repeating_action=args["repeating_action"],
        device=device,
    )

    print("[Playing] ... Environment created")

    # --- 3 - Setting up Camera ----------------------------------------------
    print("[Playing] II - Setting up Camera...")

    from isaacsim.sensors.camera import Camera

    camera_orientation = [-180.0, -90.0, 0.0]
    camera = Camera(
        prim_path="/World/camera",
        position=np.array([0.0, 0.0, 8.0]),
        orientation=euler_to_quaternion(camera_orientation, degrees=True),
        frequency=15,
        resolution=(1024, 1024)
    )
    
    camera.initialize()
    camera.add_motion_vectors_to_frame()

    print("[Playing] ... camera set up")

    output_path = "/home/g.portron/gitRepos/prex_robot_with_ultrasonic_sensors/isaac-sim/records"
    os.makedirs(output_path, exist_ok=True)

    images_path = "/home/g.portron/gitRepos/prex_robot_with_ultrasonic_sensors/isaac-sim/records/images/"
    os.makedirs(images_path, exist_ok=True)

    # --- 4 - Wrap environment ----------------------------------------------
    print("[Playing] III - Wrapping environment...")
    vec_env = DummyVecEnv([lambda: env])

    model_path = os.path.join(f"models/{args_main.model}", "ppo_prex_final")
    env_path = os.path.join(f"models/{args_main.model}", "vec_normalize.pkl")

    env = VecNormalize.load(
        env_path,
        vec_env
    )

    env.training = False
    env.norm_reward = False

    print("[Playing] ... Environment wrapped")

    # --- 5 - Create agent -------------------------------------------------
    print("[Playing] IV - Creating agent...")
    model = PPO.load(model_path, env=env, device="cpu")
    print("[Playing] ... Agent created")

    # --- 6 - Play the model -----------------------------------------------
    print("[Playing] Starting playing ...")
    
    tot_episodes = 0
    timesteps = 0

    for eps in range (args_main.nb_episodes):

        obs = env.reset()
        
        tot_episodes += 1
        done =False
        step = 0

        while not done:

            action, _ = model.predict(obs, deterministic=True)
            action = np.round(action, 4)
            linear_vel = action[0]
            angular_vel = action[1]

            obs, reward, done, info = env.step(action)

            print(f"eps = {tot_episodes}, step_count = {timesteps}, reward = {reward:.3f}, "
                    f"position = {env.position[:2]}, "
                f"lin_vel = {linear_vel}, ang_vel = {angular_vel}, distance = {env.dist:.3f}")

            camera.get_current_frame()
            rgb_image = camera.get_rgb()
            image = Image.fromarray(rgb_image)
            save_path = f"{images_path}/ep{eps}_rgb_image_{step}.png"
            image.save(save_path)

            step += 1
            timesteps += 1

### --- SAC --- ###
else:
    # --- 2 - Creating environment --------------------------------------------
    print("[Playing] I - Creating environment...")

    env = PrexIsaacEnv(
        max_episode_length=args["max_steps"],
        max_linear_speed=args["max_linear_speed"],
        max_angular_speed=args["max_angular_speed"],
        target_point=args["target_point"],
        radius_target=args["radius_target"],
        physics_dt=1.0/60.0,
        rendering_dt=1.0,
        verbose=args["verbose"],
        ppo=False,
        cube=args_main.cube,
        borderless_perimeter=args["borderless_perimeter"],
        cube_dimension=args["cube_dimension"],
        dist_objects=args["distance_between_objects"],
        arena=args_main.arena,
        arena_geometry=args["arena_geometry"],
        sensor_config=args["sensor_config"],
        repeating_action=args["repeating_action"],
        device=device,
    )

    print("[Playing] ... Environment created")

    # --- 3 - Setting up Camera ----------------------------------------------
    print("[Playing] II - Setting up Camera...")

    from isaacsim.sensors.camera import Camera

    camera_orientation = [-180.0, -90.0, 0.0]
    camera = Camera(
        prim_path="/World/camera",
        position=np.array([0.0, 0.0, 12.0]),
        orientation=euler_to_quaternion(camera_orientation, degrees=True),
        frequency=15,
        resolution=(1024, 1024)
    )

    camera.initialize()
    camera.add_motion_vectors_to_frame()

    print("[Playing] ... camera set up")

    output_path = "/home/g.portron/gitRepos/prex_robot_with_ultrasonic_sensors/isaac-sim/records"
    os.makedirs(output_path, exist_ok=True)

    images_path = "/home/g.portron/gitRepos/prex_robot_with_ultrasonic_sensors/isaac-sim/records/images/"
    os.makedirs(images_path, exist_ok=True)

    # --- 4 - Creating replay buffer ------------------------------------------
    print("[Playing] III - Creating replay buffer...")

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

    # --- 5 - Creating agent --------------------------------------------------
    print("[Playing] IV - Creating agent...")

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

    agent.load_weights(model_path)
    agent.set_to_eval_mode()

    # --- 6 - Playing -------------------------------------------------------------
    print("[Playing] Starting playing ...")

    tot_episodes = 0
    timesteps = 0

    for eps in range (args_main.nb_episodes):

        obs, _ = env.reset()

        tot_episodes += 1
        done =False
        step = 0

        while not done:
            action, _ = agent.select_action(
                torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            )
            action = action[0]

            action = np.round(action, 4)
            linear_vel = action[0]
            angular_vel = action[1]

            next_obs, reward, terminated, truncated, _ = env.step(action, render=True)
            done = terminated or truncated

            print(f"eps = {tot_episodes}, step_count = {timesteps}, reward = {reward:.3f}, "
                    f"position = {env.position[:2]}, "
                f"lin_vel = {linear_vel}, ang_vel = {angular_vel}, distance = {env.dist:.3f}")

            obs = next_obs.copy()

            camera.get_current_frame()
            rgb_image = camera.get_rgb()
            image = Image.fromarray(rgb_image)
            img_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"{images_path}/ep{eps}_rgb_image_{img_timestamp}.png"
            image.save(save_path)

            step += 1
            timesteps += 1

# --- 7 - Create video -------------------------------------------------------------
video_name = "episode_" + args_main.model + ".mp4"
video_path = os.path.join(output_path, video_name)
create_video(video_path, images_path, fps=15)

shutil.rmtree(images_path)

print(f"Command to use to get the video : scp g.portron@10.163.11.19:{video_path} Téléchargements/")

env.close()