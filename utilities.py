# Starting to setup uiltity functions to use in other places:

import gymnasium as gym
import highway_env
from matplotlib import pyplot as plt
import numpy as np
import scipy.stats as stats



def genInitialDist(numSims,env):
    numCars = 4 # TODO: need to automate later len(obs[1:])
    params = ['x','y','vx','vy'] # TODO: need to automate later len(obs[1:])

    # create dictionary
    initialStates = {}
    for param in params:
         for i in range(numCars):
            initialStates[f'car{i+1}{param}'] = np.zeros(numSims)

    
    for sim in range(numSims):
        obs, _ = env.reset()
        for car in range(numCars):
            # for car in initialState:
            # print(initialStates.keys())
            for ob in range(len(obs[car+1])):
                initialStates["car"+str(car+1)+params[ob]][sim] = obs[car+1][ob]
    return initialStates

## rollout function - take in (environment, policy, numSteps) return reward of that trajectory
def rollout(env, model, numSteps, nominalTraj, sim, numCars, params, plot):
    totalReward = 0
    obs, _ = env.reset()
    initialState = obs
    pos = []
    done = False
    for step in range(numSteps):
        # get action from trained model, make deterministic
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)


        if not done:
            # add observations for ith timestep 
            for car in range(numCars):
                for ob in range(len(obs[car])):
                    nominalTraj["car"+str(car)+params[ob]][step].append(obs[car][ob])
                    
        
        egoObs = obs[0]
        egoPosition = (egoObs[0], egoObs[1])
        pos.append(egoPosition)
        if not done:
            done = terminated or truncated
        totalReward += reward
        if plot:
            env.render()


        if done:

            # add observations for ith timestep 
            for car in range(numCars):
                for ob in range(len(obs[car])):
                    nominalTraj["car"+str(car)+params[ob]][step].append(np.nan)


            # break
    return totalReward, initialState, pos, nominalTraj


## episode function - run multiple rollouts
def episode(env, model, numSteps, numT, plot):
    rewards = []
    initStates = []
    positions = [] # list of list of (x,y) tuples
    # the following will be lists of x (float values) for each car: [numTimesteps, numTrajectories]
    
    numCars = 5 # TODO: need to automate later len(obs[])
    params = ['x','y','vx','vy'] # TODO: need to automate later 

    # create dictionary
    nominalTraj = {}
    for param in params:
         for i in range(numCars):
            nominalTraj[f'car{i}{param}'] = [[] for _ in range(numSteps)]
    print(nominalTraj.keys())

    for sim in range(numT):
        totalReward, initialState, posTrajectory, nominalTraj = rollout(env, model, numSteps, nominalTraj, sim, numCars, params, plot)
        rewards.append(totalReward)
        initStates.append(initialState)
        positions.append(posTrajectory)
    return rewards, initStates, positions, nominalTraj


# function to turn matrix of car info into a series of nominal distributions (1/timestep)                      
def nomT(matrix, save_path):
    for i, row in enumerate(matrix):
        row = np.array(row)
        mu, sigma = np.mean(row), np.std(row)
        
        # Create a range of x values for the fitted normal distribution
        x = np.linspace(min(row), max(row), 100)
        pdf = stats.norm.pdf(x, mu, sigma)  # Normal distribution PDF

        # Plot the histogram and the fitted normal curve
        plt.figure(figsize=(8, 5))
        plt.hist(row, bins=30, density=False, alpha=0.6, color='b', label="Histogram of Data")
        # plt.plot(x, pdf, 'r', linewidth=2, label="Fitted Normal Distribution")

        # Labels and title
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.title(f"Fitted Normal Distribution: μ={mu:.2f}, σ={sigma:.2f}")
        plt.legend()

        # Save the plot as an image file
        plot_filename = f"{save_path}/ego_nom_taj_distribution_{i}.png"
        plt.savefig(plot_filename, dpi=300)
        plt.close()  # Close the plot to free memory
    