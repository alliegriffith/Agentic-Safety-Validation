import gymnasium as gym
import highway_env
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy


# Load the environment
env = gym.make("roundabout-v0", render_mode="rgb_array")


# Must use the same configuration as the one you trained the model in to evalaute it
config = {
    "observation": {
        "type": "Kinematics",  # Ensures observation matches training
        "features": ["x", "y", "vx", "vy"],  # Feature set used in training
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

# Apply configuration
env.unwrapped.configure(config)
env.reset()
# Load the trained model
model = DQN.load("dqn_highway_roundabout")

# Evaluate the model over 10 episodes
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")
