# Learning highway-env
import gymnasium as gym
import highway_env
from stable_baselines3 import DQN
import matplotlib.pyplot as plt
import os
import numpy as np
import scipy.stats as stats

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
def rollout(env, policy, numSteps, ego, car1, car2, car3, car4):
    totalReward = 0
    obs, _ = env.reset()
    initialState = obs
    pos = []
    for i in range(numSteps):
        # get action from trained model, make deterministic
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        # add observations for ith timestep 
        # if there are 
        ego[i].append(obs[0][0])
        car1[i].append(obs[1][0])
        car2[i].append(obs[2][0])
        car3[i].append(obs[3][0])
        car4[i].append(obs[4][0])
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
    # the following will be lists of x (float values) for each car: [numTimesteps, numTrajectories]
    ego = [[] for _ in range(numSteps)]
    car1 = [[] for _ in range(numSteps)]
    car2 = [[] for _ in range(numSteps)]
    car3 = [[] for _ in range(numSteps)]
    car4 = [[] for _ in range(numSteps)]
    for _ in range(numT):
        totalReward, initialState, posTrajectory = rollout(env, policy, numSteps, ego, car1, car2, car3, car4)
        rewards.append(totalReward)
        initStates.append(initialState)
        positions.append(posTrajectory)
    return rewards, initStates, positions, ego, car1, car2, car3, car4

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

def showTrajectoryInfo(numT, rewards, initStates, positions, ego, car1, car2, car3, car4):
    for i in range(numT):
        print(f"for the {i}th trajectory:")
        # print("reward", rewards[i])
        print("initState", initStates[i])
        
        print("(x,y) positions of ego car")
        for z in range(len(positions[i])):
            #(x,y) position tuple
            xyTuple = positions[i][z]
            # # scale and switch y signs to align with how we want to view coords 
            # xy= ((xyTuple[0]* 100), (-xyTuple[1] * 10))
            #positions[i][z] = xyTuple
            print(xyTuple)
 
# function to turn matrix of car info into a series of nominal distributions (1/timestep)                      
def nomT(matrix):
    for i, row in enumerate(matrix):
        row = np.array(row)
        mu, sigma = np.mean(row), np.std(row)
        
        # Create a range of x values for the fitted normal distribution
        x = np.linspace(min(row), max(row), 100)
        pdf = stats.norm.pdf(x, mu, sigma)  # Normal distribution PDF

        # Create the "plots" directory if it doesn't exist
        save_path = r"C:\Users\Allie Griffith\Downloads\CS238V\finalProj\plots"

        # Plot the histogram and the fitted normal curve
        plt.figure(figsize=(8, 5))
        plt.hist(row, bins=30, density=True, alpha=0.6, color='b', label="Histogram of Data")
        plt.plot(x, pdf, 'r', linewidth=2, label="Fitted Normal Distribution")

        # Labels and title
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.title(f"Fitted Normal Distribution: μ={mu:.2f}, σ={sigma:.2f}")
        plt.legend()

        # Save the plot as an image file
        plot_filename = f"{save_path}/ego_nom_taj_distribution_{i}.png"
        plt.savefig(plot_filename, dpi=300)
        plt.close()  # Close the plot to free memory
    
## main
if __name__ == "__main__":

    # import DQN model trained using stable baselines - this is our "policy"
    model = DQN.load("dqn_highway_roundabout")
    numT = 1000
    numSteps = 80

    # run 5 trajectories, each for 10 timesteps
    rewards, initStates, positions, ego, car1, car2, car3, car4 = episode(env, model, numSteps, numT)

    # turn data into nominal trajectory distributions
    nomT(ego)
    # for i in range(numSteps):
    #     print(f"{i}th step:")
    #     ego[i]
    #     car1[i]
    # # show each trajectory info 
    # showTrajectoryInfo(numT, rewards, initStates, positions, ego, car1, car2, car3, car4)
    # # create scatter plot of x,y positions of ego car trajectories
    # #plotXvT(positions, numT)

    # print(f"Lanes Count: {env.unwrapped.config.get('lanes_count', 'Not Found')}")
    # print(f"Vehicles Count: {env.unwrapped.config.get('vehicles_count', 'Not Found')}")
    # print("Full Config:", env.unwrapped.config)
