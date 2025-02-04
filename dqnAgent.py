## File to load environment, train DQN Agent, and eval agent

import gymnasium as gym
import highway_env
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

# Create the environment
env = gym.make("roundabout-v0", render_mode="rgb_array")

# Reset environment
env.reset()

# create environment configuration
config = {
    # observation options: Kinematics, GrayscaleObservation, TimeToCollision, OccupancyGrid, Lidar, Vectorized (used in RL for batch updates)
    "observation": {
        "type": "Kinematics",  # Using kinematics observations - give position and velocity
        "features": ["x", "y", "vx", "vy"],  # Feature set
    },
    "action": {
        "type": "DiscreteMetaAction",
    },
    "lanes_count": 5,  # Number of lanes in the roundabout
    "vehicles_count": 10,  # Total number of vehicles
    "duration": 100,  # Simulation steps
    "collision_reward": -1.0,  # Penalty for collision
    "reward_speed_range": [20, 30],  # Preferred speed range
}

# Apply the configuration
env.unwrapped.configure(config)
env.reset()

# Create the DQN model - import from stable baselines
model = DQN(
    policy="MlpPolicy",  # Uses a Multi-Layer Perceptron (MLP) policy
    env=env,
    learning_rate=1e-3,  # Learning rate
    buffer_size=10000,  # Replay buffer size
    learning_starts=1000,  # Steps before training starts
    batch_size=32,  # Mini-batch size
    tau=1.0,  # Target network update parameter
    gamma=0.99,  # Discount factor
    train_freq=4,  # Train every 4 steps
    target_update_interval=100,  # Target network update frequency
    verbose=1,  # Print training details
)

# Train the model for 50,000 timesteps
model.learn(total_timesteps=50000)

# Save the trained model
model.save("dqn_highway_roundabout")

# Load the trained model
model = DQN.load("dqn_highway_roundabout")

# Evaluate the model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")

