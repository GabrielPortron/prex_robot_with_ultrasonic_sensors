import torch
from torch.distributions import MultivariateNormal
from torch.optim import Adam

import numpy as np

from network import FeedForwardNN

class PPO:
    def __init__(self, env):
        # Initialize hyperparameters
        self._init_hyperparameters()

        # Extract environment information
        self.env = env
        self.obs_dim = env.state_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        # STEP 1 
        # Initialize actor and critic networks
        self.actor = FeedForwardNN(self.obs_dim, self.act_dim)
        self.critic = FeedForwardNN(self.obs_dim, 1)

        # Initialize the optimizers
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        # Create the covariance matrix for get_action
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

    def learn(self, total_timesteps):
        t_so_far = 0 # Timesteps simulated so far

        # STEP 2
        while t_so_far < total_timesteps: 

            # STEP 3
            batch_obs, batch_acts, batch_log_probs, batch, rtgs, batch_lens = self.rollout()

            # Calculate how many timesteps we collected this batch
            t_so_far += np.sum(batch_lens)

            # Calculate V_{phi, k}
            V, _, _ = self.evaluate(batch_obs)

            # STEP 5
            # Calculate advantage
            A_k = batch_rtgs - V.detach()

            #Normalize advantages
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            step = batch_obs.size(0)
            inds = np.arange(step)
            minibatch_size = step // self.num_minibatches

            # STEP 6 & 7
            for _ in range(self.n_update_per_iteration):

                # Calculate V_phi and pi_theta(a_t | s_t)
                V, curr_log_probs, _ = self.evaluate(batch_obs, batch_acts)

                # Calculate ratios
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                # Calculate surrogate losses
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                # Calculate losses
                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V, batch_rtgs)

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

                # Learning rate annealing
                frac = (t_so_far - 1.0) / total_timesteps
                new_lr = self.lr * (1.0 - frac)
                new_lr = max(new_lr, 0.0)
                self.actor_optim.param_groups[0]["lr"] = new_lr
                self.critic_optim.param_groups[0]["lr"] = new_lr

                np.random.shuffle(inds)

                for start in range(0, step, minibatch_size):

                    end = start + minibatch_size
                    idx = inds[start:end]
                    mini_obs = batch_obs[idx]
                    mini_acts = batch_acts[idx]
                    mini_log_prob = batch_log_probs[idx]
                    mini_advantage = A_k[idx]
                    mini_rtgs = batch_rtgs[idx]
                    V, curr_log_probs, entropy = self.evaluate(mini_obs, mini_acts)
                    log_ratios = curr_log_probs - mini_log_prob
                    ratios = torch.exp(log_ratios)
                    surr1 = ratios * mini_advantage
                    surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * mini_advantage
                    actor_loss = (-torch.min(surr1, surr2)).mean()
                    entropy_loss = entropy.mean()
                    actor_loss = actor_loss - self.ent_coef * entropy_loss
                    critic_loss = nn.MSELoss()(V, mini_rtgs)


    def rollout(self):
        batch_obs = []              # batch observations
        batch_acts = []             # batch actions
        batch_log_probs = []        # log probs of each action
        batch_rews = []             # batch rewards
        batch_rtgs = []             # batch rewards-to-go
        batch_lens = []             # episodic lengths in batch
        
        # Number of timesteps run so far this batch
        t = 0

        while t < self.timesteps_per_batch:

        # Rewards this episode
        ep_rews = []

        obs = self.env.reset()
        done = False

        for ep_t in range(self.max_timesteps_per_episode):

            # Increment timesteps ran this batch so far
            t += 1

            # Collect observation
            batch_obs.append(obs)
            action, log_prob = self.get_action(obs)
            obs, rew, done, _ = self.env.step(action)
        
            # Collect reward, action, and log prob
            ep_rews.append(rew)
            batch_acts.append(action)
            batch_log_probs.append(log_prob)

            if done:
                break

        # Collect episodic length and rewards
        batch_lens.append(ep_t + 1) # plus 1 because timestep starts at 0
        batch_rews.append(ep_rews) 

        # Reshape data as tensors in the shape specified before returning
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)

        # STEP 4
        batch_rtgs = self.compute_rtgs(batch_rews)

        # Return the batch data
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def get_action(self, obs):
        mean = self.actor(obs)

        dist = MultivariateNormal(mean, self.cov_mat)

        action = dist.sample
        log_prob = dist.log_prob(action)

        return action.detach().numpy(), log_prob.detach()

    def evaluate(self, batch_obs, batch_acts):
        V = self.critic(batch_obs).squeeze()

        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)

        return V, log_probs, dist.entropy()
    
    def compute_rtgs(self, batch_rews):
        batch_rtgs = []

        for ep_rews in reversed(batch_rews):
            discounted_reward = 0

            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)
        
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

        return batch_rtgs

    def _init_hyperparameters(self):
        self.timesteps_per_batch = 4800
        self.max_timesteps_per_episode = 1600

        self.gamma = 0.95
        self.lr = 0.005

        self.n_update_per_iteration

        self.clip = 0.2