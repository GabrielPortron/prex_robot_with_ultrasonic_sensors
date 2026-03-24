import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.optim import Adam

import numpy as np

import wandb

from algorithms.model import (
    ActorNetwork,
    CriticNetwork,
)

class PPO:
    def __init__(self, 
                 env,
                 timesteps_per_batch=150,
                 max_timesteps_per_episode=300,
                 n_update_per_iteration=5,
                 lr=0.0005,
                 gamma=0.99,
                 clip=0.2,
                 lam=0.98,
                 num_minibatches=6,
                 ent_coef=0.0,
                 target_kl=0.02,
                 max_grad_norm=0.5,
                 save_freq=10,
                 deterministic=False,
                 device="cpu"
                 ):
        # Initialize algorithm hyperparameters
        self.timesteps_per_batch = timesteps_per_batch
        self.max_timesteps_per_episode = max_timesteps_per_episode
        self.n_update_per_iteration = n_update_per_iteration
        self.lr = lr
        self.gamma = gamma
        self.clip = clip
        self.lam = lam
        self.num_minibatches = num_minibatches
        self.ent_coef = ent_coef
        self.target_kl = target_kl
        self.max_grad_norm = max_grad_norm

        # Initialize miscellaneous parameters
        self.save_freq = save_freq
        self.deterministic = deterministic
        self.device = device

        # Extract environment information
        self.env = env
        self.obs_dim = self.env.state_space[0]
        self.act_dim = self.env.action_space[0]

        # Initialize actor and critic networks
        self.actor = ActorNetwork(state_dim=self.obs_dim, action_dim=self.act_dim).to(self.device)
        self.critic = CriticNetwork(state_dim=self.obs_dim, action_dim=1).to(self.device)

        # Initialize optimizers for actor and critic
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        # Initialize the covariance matrix used to query the actor for actions
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5).to(self.device)
        self.cov_mat = torch.diag(self.cov_var).to(self.device)


    def learn(self, total_timesteps):

        t_so_far = 0 # Timesteps simulated so far
        i_so_far = 0 # Iterations ran so far
        self.running_avg_reward = 0

        # STEP 2 
        while t_so_far < total_timesteps:    

            # STEP 3
            batch_obs, batch_acts, batch_log_probs, batch_rews, batch_lens, batch_vals, batch_dones = self.rollout()

            # STEP 5
            # Calculate advantage using GAE
            A_k = self.calculate_gae(batch_rews, batch_vals, batch_dones)
            V = self.critic(batch_obs).squeeze()
            batch_rtgs = A_k + V.detach()

            # Calculate how many timesteps we collected this batch
            t_so_far += np.sum(batch_lens)

            #Increment the number of iterations
            i_so_far += 1

            #Normalize advantages
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            # This is the loop where we update our network for some n epochs
            step = batch_obs.size(0)
            inds = np.arange(step)
            minibatch_size = step // self.num_minibatches
            loss = []

            # STEP 6 & 7
            for _ in range(self.n_update_per_iteration):

                # Learning rate annealing
                frac = (t_so_far - 1.0) / total_timesteps
                new_lr = self.lr * (1.0 - frac)

                # Make sure learning rate doesn't go below 0
                new_lr = max(new_lr, 0.0)
                self.actor_optim.param_groups[0]["lr"] = new_lr
                self.critic_optim.param_groups[0]["lr"] = new_lr

                # Mini-batch update
                np.random.shuffle(inds) # Shuffling the index
                for start in range(0, step, minibatch_size):
                    end = start + minibatch_size
                    idx = inds[start:end]
                    # Extract data at the sampled indices
                    mini_obs = batch_obs[idx]
                    mini_acts = batch_acts[idx]
                    mini_log_prob = batch_log_probs[idx]
                    mini_advantage = A_k[idx]
                    mini_rtgs = batch_rtgs[idx]

                    # Calculate V_phi and pi_theta(a_t | s_t)
                    V, curr_log_probs, entropy = self.evaluate(mini_obs, mini_acts)

                    # Calculate ratios
                    log_ratios = curr_log_probs - mini_log_prob
                    ratios = torch.exp(log_ratios).to(self.device)
                    approx_kl = ((ratios - 1) - log_ratios).mean()

                    # Calculate surrogate losses
                    surr1 = ratios * mini_advantage
                    surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip).to(self.device) * mini_advantage

                    # Calculate actor and critic losses
                    actor_loss = (-torch.min(surr1, surr2).to(self.device)).mean()
                    critic_loss = nn.MSELoss()(V, mini_rtgs)

                    # Entropy regularization
                    entropy_loss = entropy.mean()
                    # Discount entropy loss by given coefficient
                    actor_loss = actor_loss - self.ent_coef * entropy_loss

                    # Calculate gradients and perform backward propagation for actor network    
                    self.actor_optim.zero_grad()
                    actor_loss.backward(retain_graph=True)
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    self.actor_optim.step()

                    # Calculate gradients and perform backward propagation for critic network    
                    self.critic_optim.zero_grad()    
                    critic_loss.backward() 
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    self.critic_optim.step()

                    loss.append(actor_loss.detach())
                
                # Approximation KL Divergence
                if approx_kl > self.target_kl:
                    break
            
            # Save the model if it's time
            if i_so_far % self.save_freq == 0:
                torch.save(self.actor.state_dict(), './ppo_actor.pth')
                torch.save(self.critic.state_dict(), './ppo_critic.pth')


    def calculate_gae(self, rewards, values, dones):
        batch_advantages = [] # List to store computed advantages for each timestep

        # Iterate over each episode's reward, values and done flags
        for ep_rews, ep_vals, ep_dones in zip(rewards, values, dones):
            advantages = [] # List to store advantages for the current episode
            last_advantage = 0 # Initialize the last computed advantage

            # Calculate episode advantage in reverse order (from last timestep to first)
            for t in reversed(range(len(ep_rews))):
                if t + 1 < len(ep_rews):
                    # Calculate the temporal difference (TD) error for the current timestep
                    delta = ep_rews[t] + self.gamma * ep_vals[t+1] * (1 - ep_dones[t+1]) - ep_vals[t]
                else:
                    # Special case at the boundary (last timestep)
                    delta = ep_rews[t] - ep_vals[t]
                
                # Calculate Generalized Advantage Estimation (GAE) for the current timestep
                advantage = delta + self.gamma * self.lam * (1 - ep_dones[t]) * last_advantage
                last_advantage = advantage # Update the last advantage for the next timestep
                advantages.insert(0, advantage) # Insert advantage at the beginning of the list

            # Extend the batch_advantages list with advantages computed for the current episode
            batch_advantages.extend(advantages)
        
        # Convert the batch_advantages list to a PyTorch tensor of type float
        return torch.tensor(batch_advantages, dtype=torch.float).to(self.device)                


    def rollout(self):
        batch_obs = []              # batch observations
        batch_acts = []             # batch actions
        batch_log_probs = []        # log probs of each action
        batch_rews = []             # batch rewards
        batch_lens = []             # episodic lengths in batch
        batch_vals = []             # state values collected in batch
        batch_dones = []            # done flags collected in batch

        # Episodic data. Keeps track of rewards per episode, will get cleared upon each new episode
        ep_rews = []
        ep_vals = []
        ep_dones = []
        
        # Number of timesteps run so far this batch
        t = 0

        # Initializing variables for wandb
        running_avg_steps = 0
        eps_return = 0

        while t < self.timesteps_per_batch:

            ep_rews = []
            ep_vals = []
            ep_dones = []

            obs, _, _, _ = self.env.reset()
            done = False

            for ep_t in range(self.max_timesteps_per_episode):
                
                # Track done flag of the current state
                ep_dones.append(done)

                # Increment timesteps ran this batch so far
                t += 1

                # Track observation in this batch
                batch_obs.append(obs)

                # Calculate action and make a step in the env
                action, log_prob = self.get_action(obs)
                action = np.round(action, 2)
                val = self.critic(torch.tensor(obs, dtype=torch.float).to(self.device))

                obs, rew, _, done = self.env.step(action)

                # Calculate the average reward and print information in the console
                self.running_avg_reward = (self.running_avg_reward * ep_t + rew) / (ep_t + 1)
                print(f"eps = {self.env.episode_counter} step_count = {ep_t}, reward={rew:.3f}, runn_avg_reward={self.running_avg_reward:.3f}, distance={self.env.dist:.3f}")
                eps_return += rew

                # Collect reward, action, and log prob
                ep_rews.append(rew)
                ep_vals.append(val.flatten())
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                wandb.log(
                    {
                        "v_linear_action": action[0],
                        "v_angular_action": action[1],
                        "v_linear_robot": self.env.linear_speed,
                        "v_angular_robot": self.env.angular_speed,
                        "dist_to_center": self.env.dist,
                        "reward": rew,
                        "runn_avg_reward": self.running_avg_reward,
                        "theta": self.env.theta,
                        # "norm_derivatives_linear_speed": self.env.norm_derivatives_v,
                        # "norm_derivatives_angular_speed": self.env.norm_derivatives_w,
                        # "norm_delta_actions" : self.env.norm_delta_actions
                    }
                )

                # If the environment tells us the episode is terminated, break
                if done:
                    running_avg_steps = (running_avg_steps * (self.env.episode_counter) + self.env.step_counter) / (self.env.episode_counter + 1)

                    wandb.log(
                        {
                            "ep_return": eps_return,
                            "step_count": self.env.step_counter,
                            "avg_tot_steps": running_avg_steps,
                        }
                    )

                    eps_return = 0.0

                    break

            # Collect episodic length, rewards, state values and done flags
            batch_lens.append(ep_t + 1) # plus 1 because timestep starts at 0
            batch_rews.append(ep_rews)
            batch_vals.append(ep_vals)
            batch_dones.append(ep_dones) 

        # Reshape data as tensors in the shape specified in function description, before returning
        batch_obs = torch.tensor(batch_obs, dtype=torch.float).to(self.device)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float).to(self.device)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float).flatten().to(self.device)

        # Return the batch data
        return batch_obs, batch_acts, batch_log_probs, batch_rews, batch_lens, batch_vals, batch_dones


    def get_action(self, obs):

        # Query the actor network for a mean action
        obs = torch.tensor(obs, dtype=torch.float).to(self.device)
        mean = self.actor(obs)

        # Create a distribution with the mean action and std from the covariance matrix above
        dist = MultivariateNormal(mean, self.cov_mat)

        # Sample an action from the distribution
        action = np.round(dist.sample(), 2)

        # Calculate the log probability for that action
        log_prob = dist.log_prob(action)

        # If we're testing, just return the deterministic action.
        # Sampling should only be for training as our "exploration" factor
        if self.deterministic:
            return mean.detach().numpy(), 1
        
        # Return the sampled action and the log probability of that action in our distribution
        return action.detach().numpy(), log_prob.detach()


    def evaluate(self, batch_obs, batch_acts):

        # Query critic network for a value V for each batch_obs.
        V = self.critic(batch_obs).squeeze()

        # Calculate the log probabilities of batch actions using most recent actor network
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)

        # Return the value vector V of each observation in the batch
        # and log probabilities log_probs of each action in the batch
        return V, log_probs, dist.entropy()
