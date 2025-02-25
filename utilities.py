# Starting to setup uiltity functions to use in other places:

import gymnasium as gym
import highway_env
from matplotlib import pyplot as plt
import numpy as np



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
def rollout(env, model, numSteps):
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

def rollout_old(model,
            env,
            depth = 20,
            plot=False):

    obs, _ = env.reset()

    trajectory = []

    # Run simulation for depth steps
    for _ in range(depth):

        # TODO: print out _ later!
        action, _ = model.predict(obs,deterministic = True)
        obs, reward, done, truncated, info = env.step(action)
        experience_tuple = (obs, action, reward)
        trajectory.append(experience_tuple)
        
        if plot:
            env.render()
    
    if plot:
        # Display the rendered frame
        plt.imshow(env.render())
        plt.show()
    
    return trajectory

