import numpy as np
import wandb

from stable_baselines3.common.callbacks import BaseCallback

class PPOWandbCallback(BaseCallback):

    def __init__(self, verbose=0):
        """A Callback class that sends the information of the training to wandb when using PPO

        Args:
            verbose (int, optional): A boolean deciding if information should be given. Defaults to 0.
        """

        super().__init__(verbose)

        self._episode_count = 0
        self._running_avg_reward = 0.0
        self._running_avg_steps = 0.0
        self._current_ep_step = 0
        self._ep_return = 0.0
        self._timestep = 0.0
    
    def _on_step(self):
        """The function that sends all the information we need at each step of the training
        """

        env = self._get_raw_env()
        action = self.locals["actions"][0]
        reward = self.locals["rewards"][0]
        done = self.locals["dones"][0]

        self._ep_return += reward
        self._running_avg_reward = (self._running_avg_reward * self._timestep + reward) / (self._timestep + 1)
        self._timestep += 1
        self._current_ep_step += 1

        wandb.log({
            "linear_action": float(action[0]),
            "angular_action": float(action[1]),
            "linear_speed_robot": float(env.linear_speed),
            "angular_speed_robot": float(env.angular_speed),
            "distance_to_center": float(env.dist),
            "reward": float(reward),
            "running_average_reward": float(self._running_avg_reward)
        })

        if done:

            self._episode_count += 1
            ep_length = env.step_counter
            self._running_avg_steps = (self._running_avg_steps * (self._episode_count - 1) + ep_length) / self._episode_count
            
            wandb.log({
                "ep_return": float(self._ep_return),
                "step_count": self._current_ep_step,
                "average_tot_steps": float(self._running_avg_steps),
                "episode" : self._episode_count
            })
        
            self._ep_return = 0.0
            self._current_ep_step = 0
        return True
    
    def _on_rollout_end(self):
        """The function that sends all the information we need at the end of a rollout
        """
        
        logger_vals = self.model.logger.name_to_value
        wandb.log({
            "actor_loss": logger_vals.get("train/policy_gradient_loss", 0),
            "q_loss": logger_vals.get("train/value_loss", 0),
            "entropy": logger_vals.get("train/entropy_loss", 0),
            "approx_kl": logger_vals.get("train/approx_kl", 0),
            "clip_fraction": logger_vals.get("train/clip_fraction", 0)
        })
    
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