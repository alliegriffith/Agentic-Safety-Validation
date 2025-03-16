from int_train_ppo import PPOAgent2
from train_ppo import PPOAgent
from pipelineUtil import *
import gymnasium as gym
import highway_env
import pickle

import hydra
from omegaconf import DictConfig, OmegaConf

from plotting import plot_traj, plot_initial, heatmap_failure_spots, plot_over_time,bayesian_failure_estimation
    

## Pipeline

hRConfig = {
    "observation": {
        "type": "Kinematics",
        "features": ["x", "y", "vx", "vy"],
        "normalize": False,
        "absolute": True,
    },
    "action": {"type": "DiscreteMetaAction"},
    "lanes_count": 2,
    "vehicles_count": 5,
    "collision_reward": -2.0,
    "reward_speed_range": [20, 30],
    "policy_frequency": 15,
    "simulation_frequency": 30}

intConfig= {
    "vehicles_count": 5,  # Total vehicles in the environment
    "observation": {
        "type": "Kinematics",
        "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
        "normalize": False,
        "absolute": True,
    },
    "action": {
        "type": "DiscreteMetaAction",
        "longitudinal": True,  # Allow speed control
        "lateral": True        # Allow lane changes
    },
    "initial_vehicle_count": 5,  # Only 5 vehicles at the start
    "spawn_probability": 0.0,  # No new vehicles spawned

    # Frequency Settings
    "policy_frequency": 15,  # RL policy decision updates per second
    "simulation_frequency": 30,  # Simulation step frequency per second
    "collision_reward": -2.0,
    "normalize_reward": False
}


def pipeline(policy, environmemt, numT, numSteps, plot, verbose):
    if policy == "ppo":
        if environmemt == "roundabout":
            agent = PPOAgent(obs_dim=20, act_dim=5) 
            agent.load("data/ppo_roundabout.zip") 
            env = gym.make("roundabout-v0", render_mode="rgb_array")
            config = hRConfig
        elif environmemt == "intersection":
            agent = PPOAgent2(obs_dim=30, act_dim=5)
            agent.load("data/ppo_intersection.zip")
            env = gym.make("intersection-v1", render_mode="rgb_array")
            config = intConfig
        elif environmemt == "highway":
            agent = PPOAgent(obs_dim=20, act_dim=5)
            agent.load("data/ppo_highway.zip")
            env = gym.make("highway-v0", render_mode="rgb_array")
            config = hRConfig
        else:
            raise ValueError("Environment not found. Please enter: roundabout, intersection or highway")
        
    else:
        raise ValueError("Policy not found: ppo or dqn (coming soon!)")
    
    if verbose:
        print("Training environment:", env.spec.id)
    # using defined env and configuration, init environment
    env.unwrapped.configure(config)
    
    # run episode
    rewards, nominalTraj, fail = episodePipeline(env, agent, numSteps, numT, plot,verbose)
    
    if verbose:
        ## Direct Sampling probability of failure
        pFail = fail / numT
        print(f"Probability of failure of the {policy} agent in the {environmemt} environment is: {pFail:.4f}")  
    
        # Return Average rewards
        avgRewards = sum(rewards) / len(rewards)
        print(f"Average returns of the {policy} agent in the {environmemt} environment is: {avgRewards:.4f}")
    
    return rewards, nominalTraj, fail 

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    rewards, nominalTraj, fail = pipeline(cfg.setup.policy, cfg.setup.environment, cfg.parameters.numTraj, cfg.parameters.numSteps, cfg.debug.plot, cfg.debug.verbose)
    
    with open(f'data/data_{cfg.setup.policy}_{cfg.setup.environment}_{cfg.parameters.numTraj}_{cfg.parameters.numSteps}.pkl', 'wb') as f:
        pickle.dump(nominalTraj , f)

    # ALL Plotting
    n_cars = 5
    ind_traj = 1
    n_traj = len(nominalTraj['car0x'][0])
    plot_traj(n_cars, ind_traj, nominalTraj)
    plot_initial(n_cars, nominalTraj)
    heatmap_failure_spots(nominalTraj)
    bayesian_failure_estimation(fail, cfg.parameters.numTraj, alpha_prior=1, beta_prior=1)
    plot_over_time(nominalTraj,n_traj,cfg.setup.environment)

if __name__ == "__main__":
    main()
   