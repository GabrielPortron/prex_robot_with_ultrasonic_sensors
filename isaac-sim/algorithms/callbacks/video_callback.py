import os
import math
import shutil
import numpy as np
import wandb
from datetime import datetime
from PIL import Image

from stable_baselines3.common.callbacks import BaseCallback

from utils.utils import euler_to_quaternion, create_video

class VideoCallback(BaseCallback):
    """
    
    """

    def __init__(self,
                 output_path: str,
                 images_path: str,
                 record_every_n_episodes: int =10_000,
                 n_steps: int =150,
                 fps: int =30,
                 resolution: tuple[int, int] =(1024, 1024),
                 verbose: int =0 
                ):
        """A callback Class that takes video during a PPO training to monitor it visually

        Args:
            output_path (str): The path to make the video
            images_path (str): The path to store the images
            record_every_n_episodes (int, optional): The number of epsiodes between each record. Defaults to 10_000.
            n_steps (int, optional): The maximum number of steps of an episode. Defaults to 150.
            fps (int, optional): The number of frames per seconds for the record. Defaults to 30.
            resolution (tuple[int, int], optional): The resolution of the record. Defaults to (1024, 1024).
            verbose (int, optional): A boolean deciding if information should be given. Defaults to 0.
        """
        
        super().__init__(verbose)

        self.output_path = output_path
        self.images_path = images_path
        
        self.record_every = record_every_n_episodes
        self.n_steps = n_steps
        self.fps = fps
        self.resolution = resolution
        self._episode_count = 0
    
    def _on_training_start(self):
        """The creation and initialization of the camera at the start of the training
        """
        
        from isaacsim.sensors.camera import Camera

        self.camera = Camera(
            prim_path="/World/camera",
            position=np.array([0.0, 0.0, 5.0]),
            orientation=euler_to_quaternion([-180.0, -90.0, 0.0], degrees=True, inverted=True),
            frequency=self.fps,
            resolution=self.resolution
        )

        self.camera.initialize()
        print("[VideoCallback] Camera initialised.")

    def _on_step(self):
        """A function that browses across the episodes and start the record when it's time 
        (depending on the parameter record_every_n_episodes)
        """

        done = self.locals["dones"][0]

        if done:
            self._episode_count += 1
            if self._episode_count % self.record_every == 0:
                self._record_episode()

        return True
    
    def _record_episode(self):
        """The function that records the episode
        """
        env = self._get_raw_env()

        obs = self.training_env.reset()
        done = False
        step = 0

        os.makedirs(self.images_path, exist_ok=True)

        for _ in range(10):
            env.world.step(render=True)
            self.camera.get_current_frame() 

        for _ in range(self.n_steps):

            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, done, _ = self.training_env.step(action)

            env.world.step(render=True)

            self.camera.get_current_frame()
            rgb = self.camera.get_rgb()
            image = Image.fromarray(rgb)
            save_path = os.path.join(self.images_path, f"ep{self._episode_count}_rgb_{step:04d}.png")
            image.save(save_path)

            step += 1

            if done[0]:
                break
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = "episode_" + timestamp + ".mp4"
        video_path = os.path.join(self.output_path, video_name)

        create_video(video_path, self.images_path, self.fps)
        shutil.rmtree(self.images_path)

        print("[VideoCallback] Done. ")
        print(f"Command to use to get the video : scp g.portron@10.163.11.19:{video_path} Téléchargements/")
    
    def _get_raw_env(self):
        """The function that allows to unwrap the environment

        Returns:
            env: The raw environment with no wrappers
        """
        env = self.training_env

        if hasattr(env, "venv"):
            env = env.venv
        if hasattr(env, "envs"):
            env = env.envs[0]
        if hasattr(env, "env"):
            env = env.env

        return env