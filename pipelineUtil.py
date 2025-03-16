# Utility functions for pipeline
# Learning highway-env
import gymnasium as gym
import highway_env
from stable_baselines3 import DQN
import matplotlib.pyplot as plt
import os
import numpy as np
import scipy.stats as stats
from collections import Counter
from sklearn.mixture import GaussianMixture
import torch
from int_train_ppo import PPOAgent2
from train_ppo import PPOAgent
from gymnasium.wrappers import RecordVideo

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## rollout function - take in (environment, policy, numSteps) return reward of that trajectory
def rolloutPipeline(env, model, numSteps, nominalTraj, numCars, params, plot):
    fail = 0
    totalReward = 0
    obs, _ = env.reset()
    obs = torch.tensor(obs.flatten(), dtype=torch.float32, device=device)
    done = False
    for step in range(numSteps):
        # get action from trained model, make deterministic
        action = model.act(obs, return_log_prob=False)
        obs, reward, terminated, truncated, info = env.step(int(action[0]))

        # add observations for ith timestep
        if not done:
            for car in range(numCars):
                for ob in range(len(obs[car])):
                    nominalTraj["car"+str(car)+params[ob]][step].append(obs[car][ob])
            # re-check if done now
            done = terminated or truncated
        
        if terminated and info.get("crashed", False):  
            fail = 1
            
        totalReward += reward
        
        if plot:
            env.render()

        if done:
            # if done, fill in the rest of the nominal trajectory data with nan
            for car in range(numCars):
                for ob in range(len(obs[car])):
                    nominalTraj["car"+str(car)+params[ob]][step].append(np.nan)
        
        # flatten obs after saving data
        obs = torch.tensor(obs.flatten(), dtype=torch.float32, device=device)
    return totalReward, nominalTraj, fail

## episode function - run multiple rollouts
def episodePipeline(env, model, numSteps, numT, plot, verbose):
    rewards = []
    numCars = 5 
    numFail = 0
    
    if env.spec.id == "highway-v0" or env.spec.id == "roundabout-v0":
        params = ['x','y','vx','vy'] 

    if env.spec.id == "intersection-v1":
        params = ['x','y','vx','vy','cos_h','sin_h']
        
    # create dictionary
    nominalTraj = {}
    for param in params:
         for i in range(numCars):
            nominalTraj[f'car{i}{param}'] = [[] for _ in range(numSteps)]
            
    for i in range(numT):
        totalReward, nominalTraj, fail = rolloutPipeline(env, model, numSteps, nominalTraj, numCars, params, plot)
        rewards.append(totalReward)
        numFail += fail
        if verbose:
            if i % 25 == 0:
                print(f"trajectory: {i}")
    return rewards, nominalTraj, numFail
