# ## Code to just load and simulate the Highway Env
# can use to get simulation data and self-train a model if want
import gymnasium as gym
import highway_env
from matplotlib import pyplot as plt

# Create the roundabout environment
env = gym.make("roundabout-v0", render_mode="rgb_array")
env.reset()

#  customize environment - todo

# D is list of experience tuples - POMDP
D = []

# Run simulation for 3 steps
for _ in range(20):
    action = env.unwrapped.action_type.actions_indexes["IDLE"]
    obs, reward, done, truncated, info = env.step(action)
    experience_tuple = (obs, reward)
    D.append(experience_tuple)
    
    env.render()

# Display the rendered frame
plt.imshow(env.render())
plt.show()

# messy - try new env config for more informative obs
for experience_tuple in D:
    print("observation", experience_tuple[0])
    print("reward", experience_tuple[1])

