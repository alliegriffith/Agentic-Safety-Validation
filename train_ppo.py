# New PPO Code
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
import gymnasium as gym
import highway_env
import numpy as np
import os
import zipfile


# Load the environment
env = gym.make("roundabout-v0", render_mode="rgb_array")

# Configure environment
config = {
    "observation": {
        "type": "Kinematics",
        "features": ["x", "y", "vx", "vy"],
    },
    "action": {"type": "DiscreteMetaAction"},
    "lanes_count": 2,
    "vehicles_count": 5,
    "duration": 100,
    "collision_reward": -1.0,
    "reward_speed_range": [20, 30],
    "policy_frequency": 15,
    "simulation_frequency": 30,
}

env.unwrapped.configure(config)
obs, _ = env.reset()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
import gymnasium as gym
import highway_env
import numpy as np
import os
import zipfile

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# init policy network - agent 
class PolicyNetwork(nn.Module):
    def __init__(self, input_size=20, output_size=5, hidden_size=64):
        # neetwork is a feedforward neural network with 2 hidden layers
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
        # softmax activation function - makes agent give probability distribution over actions? -> do later
        #self.softmax = nn.Softmax(dim=-1)

    # run obs through network and return logits
    def forward(self, obs):
        logits = self.network(obs)
        return logits
        #return self.softmax(logits)

# init critic network - estimates value of state
class CriticNetwork(nn.Module):
    def __init__(self, input_size=20, hidden_size=64):
        # network is a feedforward neural network with 2 hidden layers
        super(CriticNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    # returns V(s)
    def forward(self, obs):
        return self.network(obs).squeeze(-1)

# init PPO agent
class PPOAgent:
    # init both policies and use adama optimizers for both
    def __init__(self, obs_dim, act_dim, lr=3e-4, critic_lr=1e-3):
        self.policy = PolicyNetwork(obs_dim, act_dim).to(device)
        self.critic = CriticNetwork(obs_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
    
    # given obs and policy, returns action, log prob of action, and value of state
    def act(self, observations, return_log_prob=False):
        observations = observations.flatten().to(device)
        action_logits = self.policy(observations)
        value = self.critic(observations)
        
        # if in training mode, non-deterministics so can explore
        if return_log_prob: 
            action_dist = distributions.Categorical(logits=action_logits)  
            sampled_action = action_dist.sample()
            log_prob = action_dist.log_prob(sampled_action)
            return sampled_action.item(), log_prob.item(), value.item()
        
        # if in evaluation mode, deterministic so return max action prob
        else:
            action_probs = torch.softmax(action_logits, dim=-1)
            return torch.argmax(action_probs).item(), None, value.item()

    # update policy and critic networks
    def update(self, obs, old_actions, old_log_probs, advantages, returns, clip_epsilon=0.2):
        # turn all input to tensors and move to device for pytorch operations
        #obs = torch.stack([torch.tensor(o, dtype=torch.float32) for o in obs]).to(device)
        obs = torch.tensor(np.array(obs), dtype=torch.float32, device=device)
        #obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
        old_actions = torch.tensor(old_actions, dtype=torch.long).to(device)
        #old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32).to(device)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32, device=device).detach()
        advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)

        with torch.no_grad():
            action_logits = self.policy(obs)
            dist = distributions.Categorical(logits=action_logits)
            new_log_probs = dist.log_prob(old_actions)

        
        # get the ratio of the new policy to the old policy
        z = torch.exp(new_log_probs - old_log_probs)
        # ppo loss function
        clipped_ratio = torch.clamp(z, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
        policy_loss = -torch.min(z * advantages, clipped_ratio * advantages).mean()
        values = self.critic(obs)
        
        # value loss function
        value_loss = nn.MSELoss()(values, returns)

        # back prop ppo policy and value loss to update newtork weights
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

    # function to save best model during training
    def save(self, path="ppo_best_model2.zip"):
        checkpoint = {
            "policy_state_dict": self.policy.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "policy_optimizer_state_dict": self.optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
        }
        torch.save(checkpoint, "ppo_temp.pth")
        with zipfile.ZipFile(path, 'w') as zipf:
            zipf.write("ppo_temp.pth")
        os.remove("ppo_temp.pth")
        print(f"Best model saved to {path}")

# rollout function to collect data from environment
def rollout(env, agent, num_steps=80):
    observations, actions, rewards, log_probs, values = [], [], [], [], []
    obs, _ = env.reset()
    obs_t = torch.tensor(obs.flatten(), dtype=torch.float32, device=device)

    for _ in range(num_steps):
        action, log_prob, val = agent.act(obs_t, return_log_prob=True)
        observations.append(obs_t.cpu().numpy())
        actions.append(action)
        log_probs.append(log_prob)
        values.append(val)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)

        obs_t = torch.tensor(next_obs.flatten(), dtype=torch.float32, device=device)
        if terminated or truncated:
            break
    
    return observations, actions, rewards, log_probs, values

# using GAE to compute advantages and returns
def compute_advantages(rewards, values, gamma=0.99, lam=0.95):
    # add a zero to the end of values to compute deltas
    values = np.append(values, 0.0)  
    # list of advantages, one per time step
    advantages = np.zeros_like(rewards, dtype=np.float32)
    # init gae to 0
    gae = 0.0

    # move backwards through rewards and values to compute advantages
    for t in reversed(range(len(rewards))):
        # delta = r_t + gamma * V(s_{t+1}) - V(s_t)
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        # gae = delta + gamma * lambda * gae
        gae = delta + gamma * lam * gae
        # store gae in list
        advantages[t] = gae

    returns = values[:-1] + advantages
    return advantages, returns

def train_ppo(env, num_epochs=1000, batch_size=32, num_steps=80, K_epochs=10, minibatch_size=256):
    # ensure obs_dim and act_dim match agent/environment
    obs_dim = 20
    act_dim = 5
    agent = PPOAgent(obs_dim, act_dim)

    best_total_reward = -float('inf')  # Track the highest total reward

    # loop through epochs
    for epoch in range(num_epochs):
        # init lists to store data
        batch_obs, batch_actions, batch_rewards, batch_log_probs, batch_values = [], [], [], [], []
        
        # perform batch_size number of rollouts, collecting all data
        for _ in range(batch_size):
            obs, actions, rewards, log_probs, values = rollout(env, agent, num_steps)
            batch_obs.extend(obs)
            batch_actions.extend(actions)
            batch_rewards.extend(rewards)
            batch_log_probs.extend(log_probs)
            batch_values.extend(values)

        # compute advantages (for ppo agent) and returns (for critic)
        advantages, returns = compute_advantages(batch_rewards, batch_values)

        dataset_size = len(batch_obs)
        indices = np.arange(dataset_size)
        
        # perform multiple (10) gradient updates on batch of data
        for _ in range(K_epochs):
            # randomly select 256 steps of data
            np.random.shuffle(indices)
            start = 0
            while start < dataset_size:
                end = start + minibatch_size
                # crate minibatch
                mb_idx = indices[start:end]
                mb_obs = [batch_obs[i] for i in mb_idx]
                mb_actions = [batch_actions[i] for i in mb_idx]
                mb_log_probs = [batch_log_probs[i] for i in mb_idx]
                mb_advantages = [advantages[i] for i in mb_idx]
                mb_returns = [returns[i] for i in mb_idx]

                # update agent with minibatch
                agent.update(mb_obs, mb_actions, mb_log_probs, mb_advantages, mb_returns)
                # done with minibatch update, move to next
                start = end

        # calc and return rewards of epoch
        total_reward = np.sum(batch_rewards)
        average_reward = total_reward / batch_size
        print(f"Epoch {epoch+1}/{num_epochs} - Total Reward: {total_reward:.2f} - Average Reward: {average_reward:.2f}")

        # save best model as train
        if total_reward > best_total_reward:
            best_total_reward = total_reward
            agent.save("ppo_best_model.zip")
            print(f"New best model saved with total reward {best_total_reward:.2f}!")

    return agent

# train PPO agent
#ppo_agent = train_ppo(env, num_epochs=1000, batch_size=32, num_steps=80, K_epochs=10, minibatch_size=256)

# # Save trained model
# ppo_agent.save("trained_ppo_agent.zip")

# PPO Agent Class for intersection agent
# PPO Policy Wrapper
class PPOAgent2:
    # init behavior policy, critic and adam optimizer
    def __init__(self, obs_dim, act_dim, lr=3e-4, critic_lr=1e-3):
        self.policy = PolicyNetwork(obs_dim, act_dim).to(device)
        self.critic = CriticNetwork(obs_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        
    # choose action to take 
    def act(self, observations, return_log_prob=False):
        # get observation matrix [4,5], flatten so its a 1D tensor (already a tensor)
        observations = observations.flatten()

        # get prob distrib over actions by running obs through our nueral net policy
        action_probs = self.policy(observations)
        value = self.critic(observations)  
        
        # if return_log_prob, in training mode -> return sampled actions
        if return_log_prob:
            # action_dist is a scalar tensor
            action_dist = distributions.Categorical(action_probs)
            sampled_action = action_dist.sample()
            log_prob = action_dist.log_prob(sampled_action)
            return sampled_action.item(), log_prob.item(), value.cpu().item()
        # else, deterministic so return best action
        return torch.argmax(action_probs).item(), None, value.item()


    def update(self, obs, old_actions, old_log_probs, advantages, returns, clip_epsilon=0.2):
        # Policy update with PPO- Clipped loss objective
        obs = torch.stack([torch.tensor(o, dtype=torch.float32) for o in obs]).to(device)
        # stack observations (already flattened tensors)
        #obs = torch.stack(obs).to(device)
        # first turn all input components into tensors
        old_actions = torch.tensor(old_actions, dtype=torch.long).to(device)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32).to(device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)

        # Compute new log probabilities
        # get action distribs for obs under new policy -> log probs for eqn
        action_probs = self.policy(obs)
        action_dist = distributions.Categorical(action_probs)
        new_log_probs = action_dist.log_prob(old_actions)
        values = self.critic(obs)
        
        # normalize advantages for more stability 
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Compute the probability ratio - subtract instead of divide bc log probs
        z = torch.exp(new_log_probs - old_log_probs)

        # Compute the clipped objective function =>  1-e < z < 1 + e
        clipped_ratio = torch.clamp(z, 1 - clip_epsilon, 1 + clip_epsilon)
        # take mean loss (over all trajectories in batch) 
        # flip sign bc performing gradient ascent
        loss = -torch.min(z * advantages, clipped_ratio * advantages).mean()
        
        # calc loss for critic network too (mean squared error)
        #value_loss = nn.MSELoss()(values, returns)
        value_loss = nn.MSELoss()(values, torch.tensor(returns, dtype=torch.float32, device=device))


        # backprop loss to update policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # backprop critic loss to optimize critic seperately
        # Optimize critic separately
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()
        
    def save(self, path="ppo_agent.zip"):
        """Save model weights, optimizers, and training parameters to a zip file."""
        checkpoint = {
            "policy_state_dict": self.policy.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "policy_optimizer_state_dict": self.optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
        }

        torch.save(checkpoint, "ppo_temp.pth")

        # Compress into a ZIP file
        with zipfile.ZipFile(path, 'w') as zipf:
            zipf.write("ppo_temp.pth")
        
        # Remove the temporary file after saving
        os.remove("ppo_temp.pth")

        print(f"Agent saved successfully to {path}")

    def load(self, path="ppo_agent.zip", path2 = "ppo_temp.pth"):
        """Load model weights, optimizers, and training parameters from a zip file."""
        # Extract ZIP
        with zipfile.ZipFile(path, 'r') as zipf:
            zipf.extractall()

        checkpoint = torch.load(path2, map_location=device)

        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.optimizer.load_state_dict(checkpoint["policy_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])

        # Remove extracted file after loading
        os.remove(path2)

        print(f"Agent loaded successfully from {path}")
