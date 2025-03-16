import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import highway_env
import numpy as np
import random
import os
import zipfile
from collections import deque
import csv

# set device, adaptable to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Q newtork class uses nn to estimate q-val
class QNetwork(nn.Module):
    # init simple sequential neural net with relu activations
    def __init__(self, input_size=20, output_size=5, hidden_size=64):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

# create replay buffer to store experience tuples 
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        # using a double-ended que for the replay buffer so dynamically get rid of old samples
        # when reach capacity (from left)
        self.buffer = deque(maxlen=capacity)
    
    # create method that randomly samples batch_size experience tuples from the buffer
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        # seperate the values from each experience tuple in batchx, turn into numpy array
        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch], dtype=np.float32)
        next_states = np.array([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch], dtype=np.float32)
        
        return states, actions, rewards, next_states, dones


# DQN agent class - used for safety validation pipeline
class DQNAgent:
    def __init__(self, obs_dim, act_dim, hidden_size=64, lr=1e-3, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=500, buffer_capacity=10000):
        # init basic values/hyperparams
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0
        
        # init main q network and target q network
        self.q_network = QNetwork(obs_dim, act_dim, hidden_size).to(device)
        self.target_network = QNetwork(obs_dim, act_dim, hidden_size).to(device)
        # .load_state_dict() loads same model weights and biases as main q network so they match 
        # cite - https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.load_state_dict
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # adam optimizer 
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
    
    def select_action(self, state):
        # e-greedy epsilon selection
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                  np.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        # randomly select action epsilon % of time
        if random.random() < epsilon:
            return random.randrange(self.act_dim)
        # else select action with highest q value from main q network
        else:
            state_tensor = torch.from_numpy(state).float().to(device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def update(self, batch_size):

        # sample minibatch of experience tuples from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # convert nupy arrays to tensors
        states = torch.from_numpy(states).float().to(device)
        actions = torch.from_numpy(actions).long().to(device)
        rewards = torch.from_numpy(rewards).float().to(device)
        next_states = torch.from_numpy(next_states).float().to(device)
        dones = torch.from_numpy(dones).float().to(device)
        
        # calc q values for all states in batch
        q_values = self.q_network(states)
        # only keep q values for actions actually took
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # calc target q vals with Q_hat = r + y*maxa'(Q(s',a')) (Bellman target)
        with torch.no_grad():
            # get all Q values (all actions for all next states)
            next_q_values = self.target_network(next_states)
            # take only Q-val for max action
            max_next_q_values = next_q_values.max(1)[0]
            # calc target q val with Bellman target eqn, if terminal set to 0
            expected_q_value = rewards + self.gamma * max_next_q_values * (1 - dones)
        
        # loss is mean squared error btwn q est and q target
        loss = nn.MSELoss()(q_value, expected_q_value)
        
        # optimize main q target
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        # copy main q networks' weights and biases
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    # function to save model params to a zip file so can be used in pipeline
    def save(self, path):
        checkpoint = {
            "q_network_state_dict": self.q_network.state_dict(),
            "target_network_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(checkpoint, "dqn_temp.pth")
        with zipfile.ZipFile(path, 'w') as zipf:
            zipf.write("dqn_temp.pth")
        os.remove("dqn_temp.pth")
        print(f"Best model saved to {path}")

# dqn training loop
def train_dqn(env, num_episodes=1000, batch_size=64, target_update=10, max_steps=100):
    obs_dim = 20   # 5 vehicles * 4-6 features per vehicle (x, y, vx, vy, cos_h, sin_h)
    act_dim = 5    # 5 actions always
    agent = DQNAgent(obs_dim, act_dim)
    best_reward = -float('inf')
    
    # list to store episode metrics for CSV output
    episode_metrics = []
    # loop through each episode
    for episode in range(num_episodes):
        obs, _ = env.reset()
        obs = obs.flatten()  
        total_reward = 0
        losses = []
        
        for step in range(max_steps):
            # policy selects action with e-greedy
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_obs = next_obs.flatten()
            done = terminated or truncated
            
            # add experience tuple to replay buffer
            agent.replay_buffer.buffer.append(obs, action, reward, next_obs, done)
            
            obs = next_obs
            total_reward += reward
            
            # update Q-network w/ minibatch from replay buffer
            loss_val = agent.update(batch_size)
            if loss_val is not None:
                losses.append(loss_val)
            # if terminated or truncted, break from episode
            if done:
                break
        
        # every 10 episodes update target network to match policy network
        if episode % target_update == 0:
            agent.update_target_network()
        
        # avg loss for episode
        avg_loss = np.mean(losses) if losses else 0.0
        
        # epsilon decays over time
        current_epsilon = agent.epsilon_end + (agent.epsilon_start - agent.epsilon_end) * \
                          np.exp(-1. * agent.steps_done / agent.epsilon_decay)
        
        print(f"Episode {episode+1}/{num_episodes} - Total Reward: {total_reward:.2f} - Steps: {step+1} - Avg Loss: {avg_loss:.4f} - Epsilon: {current_epsilon:.4f}")
        
        # Save best model based on episode reward
        if total_reward > best_reward:
            best_reward = total_reward
            agent.save("dqn_best_highway2crNew.zip")
            print(f"New best model saved with total reward {best_reward:.2f}!")
        
        # Append metrics for this episode
        episode_metrics.append({
            'episode': episode + 1,
            'total_reward': total_reward,
            'steps': step + 1,
            'avg_loss': avg_loss,
            'epsilon': current_epsilon
        })
    
    # Save metrics to CSV file
    csv_filename = 'dqn_highway_metrics.csv'
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['episode', 'total_reward', 'steps', 'avg_loss', 'epsilon']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in episode_metrics:
            writer.writerow(row)
    print(f"Training metrics saved to {csv_filename}")
    
    return agent

if __name__ == "__main__":
        
    env = gym.make("highway-v0", render_mode="rgb_array")

    # Intersection configuration
    config= {
        "observation": {
            "type": "Kinematics",
            "features": ["x", "y", "vx", "vy"],
        },
        "action": {"type": "DiscreteMetaAction"},
        "lanes_count": 2,
        "vehicles_count": 5,
        "collision_reward": -2.0,
        "reward_speed_range": [20, 30],
        "policy_frequency": 15,
        "simulation_frequency": 30}

    env.unwrapped.configure(config)
    obs, _ = env.reset()

    # train agent
    agent = train_dqn(env, num_episodes=1000, batch_size=64, target_update=10, max_steps=100)

# intersection config
# config= {
#     "vehicles_count": 5,  # Total vehicles in the environment
#     "observation": {
#         "type": "Kinematics",
#         "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
#     },
#     "action": {
#         "type": "DiscreteMetaAction",
#         "longitudinal": True,  # Allow speed control
#         "lateral": True        # Allow lane changes
#     },
#     "initial_vehicle_count": 5,  # Only 5 vehicles at the start
#     "spawn_probability": 0.0,  # No new vehicles spawned

#     # Frequency Settings
#     "policy_frequency": 15,  # RL policy decision updates per second
#     "simulation_frequency": 30,  # Simulation step frequency per second
#     "collision_reward": -2.0,
#     "normalize_reward": False
# }


# env.unwrapped.configure(config)
# obs, _ = env.reset()

# # train agent
# agent = train_dqn(env, num_episodes=1000, batch_size=64, target_update=10, max_steps=100)

# DQN class for intersection DQN agent -- add class for intersection dimesions for easy pipeline config
class DQNAgent2:
    def __init__(self, obs_dim, act_dim, hidden_size=64, lr=1e-3, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=500, buffer_capacity=10000):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0
        
        # Q-Network and Target Network
        self.q_network = QNetwork(obs_dim, act_dim, hidden_size).to(device)
        self.target_network = QNetwork(obs_dim, act_dim, hidden_size).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
    
    def act(self, state, training=True):
        # if training = false, make deterministic/greedy
        if not training:
            epsilon = 0
            state_tensor = state
        else:
            epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                      np.exp(-1. * self.steps_done / self.epsilon_decay)
            state_tensor = torch.from_numpy(state).float().to(device)
        
        self.steps_done += 1
        if random.random() < epsilon:
            return random.randrange(self.act_dim)
        else:
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def update(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return
        
        # Sample a minibatch of transitions
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # Convert to tensors
        states = torch.from_numpy(states).float().to(device)
        actions = torch.from_numpy(actions).long().to(device)
        rewards = torch.from_numpy(rewards).float().to(device)
        next_states = torch.from_numpy(next_states).float().to(device)
        dones = torch.from_numpy(dones).float().to(device)
        
        # Compute current Q-values: Q(s, a)
        q_values = self.q_network(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values using the target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            max_next_q_values = next_q_values.max(1)[0]
            expected_q_value = rewards + self.gamma * max_next_q_values * (1 - dones)
        
        # Compute loss (MSE)
        loss = nn.MSELoss()(q_value, expected_q_value)
        
        # Optimize the Q-network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save(self, path="dqn_best.zip"):
        checkpoint = {
            "q_network_state_dict": self.q_network.state_dict(),
            "target_network_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(checkpoint, "dqn_temp.pth")
        with zipfile.ZipFile(path, 'w') as zipf:
            zipf.write("dqn_temp.pth")
        os.remove("dqn_temp.pth")
        print(f"Best model saved to {path}")
        
    def load(self, path="dqn_best.zip"):
        temp_pth_path = "dqn_temp.pth"
        
        # Unzip the model checkpoint
        with zipfile.ZipFile(path, 'r') as zipf:
            zipf.extract(temp_pth_path)
        
        # Load the saved checkpoint
        checkpoint = torch.load(temp_pth_path, map_location=device)
        
        # Load state dictionaries into existing networks and optimizer
        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Move models back to the correct device
        self.q_network.to(device)
        self.target_network.to(device)
        
        # Remove temporary file
        os.remove(temp_pth_path)
        
        print(f"Model loaded successfully from {path}")
