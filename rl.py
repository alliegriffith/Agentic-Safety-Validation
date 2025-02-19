# Learning highway-env
import gymnasium as gym
import highway_env
from stable_baselines3 import DQN
import matplotlib.pyplot as plt
import os

# Load the environment
env = gym.make("roundabout-v0", render_mode="rgb_array")

# create configuration
config = {
    "observation": {
        "type": "Kinematics",  
        "features": ["x", "y", "vx", "vy"],  
    },
    "action": {
        # if want more control over action space, switch to "ContinuousAction"
        "type": "DiscreteMetaAction",
    },
    "lanes_count": 2,  # Number of lanes in the roundabout - fixed
    "vehicles_count": 5,  # fixed
    "duration": 100,  
    "collision_reward": -1.0,  
    "reward_speed_range": [20, 30],  
    # policy_frequency = agent makes decisions 5 times per second
    "policy_frequency": 15,   # increase to make the agent act more frequently
    # environment updates 30 times per second
    "simulation_frequency": 30,  # Increase to make environment update more frequently
    # agent makes a decision every simulation_frequency / policy_frequency steps.
}

env.unwrapped.configure(config) 

## rollout function - take in (environment, policy, numSteps) return reward of that trajectory
def rollout(env, policy, numSteps):
    totalReward = 0
    obs, _ = env.reset()
    initialState = obs
    pos = []
    for _ in range(numSteps):
        # get action from trained model, make deterministic
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        egoObs = obs[0]
        egoPosition = (egoObs[0], egoObs[1])
        pos.append(egoPosition)
        done = terminated or truncated
        totalReward += reward
        env.render()
        if done:
            break
    return totalReward, initialState, pos

## episode function - run multiple rollouts
def episode(env, policy, numSteps, numT):
    rewards = []
    initStates = []
    positions = [] # list of list of (x,y) tuples
    for _ in range(numT):
        totalReward, initialState, posTrajectory = rollout(env, policy, numSteps)
        rewards.append(totalReward)
        initStates.append(initialState)
        positions.append(posTrajectory)
    return rewards, initStates, positions

def plotXvT(positions, numT):
    save_path = r"C:\Users\Allie Griffith\Downloads\CS238V\finalProj\plots"
    # Define full file path for saving
    plot_file = os.path.join(save_path, "XvTPlot.png")
    # Plot x position over time
    plt.figure(figsize=(8, 6))

    for i in range(numT):
        x_vals = [xy[0] * 100 for xy in positions[i]]
        time_steps = list(range(len(x_vals)))

        plt.plot(time_steps, x_vals, marker="o", linestyle="-", alpha=0.7, label=f"Trajectory {i}")

    # Formatting the time-series plot
    plt.xlabel("Time Step")
    plt.ylabel("X Position (scaled)")
    plt.title("Ego Car X Position Over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_file)
    plt.close()
    
def plotXvY(positions, numT):
    save_path = r"C:\Users\Allie Griffith\Downloads\CS238V\finalProj\plots"
    # Define full file path for saving
    plot_file = os.path.join(save_path, "XvYPlot.png")
    # Plot all ego car trajectories in the roundabout
    plt.figure(figsize=(8, 6))

    # Iterate through each trajectory
    for i in range(numT):
        x_vals = [xy[0] * 100 for xy in positions[i]]  # Scale x positions
        y_vals = [-xy[1] * 10 for xy in positions[i]]  # Scale and flip y positions for visualization
        
        plt.plot(x_vals, y_vals, marker="o", linestyle="-", alpha=0.7, label=f"Trajectory {i}")

    # Formatting the plot
    plt.xlabel("X Position (scaled)")
    plt.ylabel("Y Position (scaled and flipped)")
    plt.title("Ego Car Trajectories in the Roundabout")
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_file)
    plt.close()

def showTrajectoryInfo(numT, rewards, initStates, positions):
    for i in range(numT):
        print(f"for the {i}th trajectory:")
        print("reward", rewards[i])
        print("initState", initStates[i])
        # print("(x,y) positions of ego car")
        # for z in range(len(positions[i])):
        #     #(x,y) position tuple
        #     xyTuple = positions[i][z]
        #     # scale and switch y signs to align with how we want to view coords 
        #     xy= ((xyTuple[0]* 100), (-xyTuple[1] * 10))
        #     #positions[i][z] = xyTuple
        #     print(xy)
            
            
## main
# import DQN model trained using stable baselines - this is our "policy"
model = DQN.load("dqn_highway_roundabout")
numT = 10
numSteps = 100

# run 5 trajectories, each for 10 timesteps
rewards, initStates, positions = episode(env, model, numSteps, numT)

# show each trajectory info 
showTrajectoryInfo(numT, rewards, initStates, positions)
# create scatter plot of x,y positions of ego car trajectories
#plotXvT(positions, numT)

# print(f"Lanes Count: {env.unwrapped.config.get('lanes_count', 'Not Found')}")
# print(f"Vehicles Count: {env.unwrapped.config.get('vehicles_count', 'Not Found')}")
# print("Full Config:", env.unwrapped.config)
