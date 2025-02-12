# Starting to setup uiltity functions to use in other places:

import gymnasium as gym
import highway_env
from matplotlib import pyplot as plt


def rollout(model,
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

