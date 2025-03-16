
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

import seaborn as sns

import pickle

import scipy.stats as stats


def plot_traj(n_cars, ind_traj, traj_lists):
    """
    Plot all cars in one trajectory rollout

    Input: 
    n_cars - Number of cars
    ind_traj - Trajectory index in stored dataset of trajectories
    traj_lists - All stored trajectories in list
    """

    for i in range(n_cars):
        x = np.array([list[ind_traj] for list in traj_lists['car'+str(i)+'x']])  # Your full x array
        y = np.array([list[ind_traj] for list in traj_lists['car'+str(i)+'y']])*-1  # Your full x array

        plt.plot(x, y)
    return x


def plot_initial(n_cars, traj_lists):
    """
    Plot all initial points of cars from a list of trajectories

    Input: 
    n_cars - Number of cars in one rollout
    traj_lists - All stored trajectories in list
    """

    # Figure out which indicies are nan for figuring out if there was a collision:
    # nan_indices = np.isnan( np.array(traj_lists['car0x'][-1][0:-4]))
    nan_indices = np.isnan( np.array(traj_lists['car0x'][-1][0:len(traj_lists['car0x'][0])]))
 
    fig, axs = plt.subplots(1,2)
    fig.suptitle('Initial Positions of Cars')

    # Create the plot
    for i in range(n_cars):
        x = np.array(traj_lists['car'+str(i)+'x'][0])  # 0 for initial point
        y = np.array(traj_lists['car'+str(i)+'y'][0])*-1

        axs[0].scatter(x,y, label = 'car'+str(i)+'x', alpha = 0.2,s = 10)
        axs[1].scatter(x[nan_indices], y[nan_indices], label=f'car{i} Failed', alpha=0.2, s=10)


    # non failures
    axs[0].set_title("All Trajectories")
    axs[0].set_xlabel("X Position")
    axs[0].set_ylabel("Y Position")

    # failures
    axs[1].set_title("Failed Trajectories")
    axs[1].set_xlabel("X Position")
    axs[1].set_ylabel("Y Position")

    # axs[0].gca().set_aspect('equal', adjustable='box')
    # axs[1].gca().set_aspect('equal', adjustable='box')

    axs[0].legend()
    axs[1].legend()

    axs[0].grid()
    axs[1].grid()
    plt.show()


def animate(i,fig,nan_num_prev,traj_lists,num_traj,environment ):
    fig.clear()

    for car in range(5):
        x = np.array(traj_lists['car'+str(car)+'x'][i]) 
        y = np.array(traj_lists['car'+str(car)+'y'][i])*-1


        if environment == "highway":
            # non failures
            plt.scatter(x,y, label = 'car'+str(car), alpha = 0.3)
        else:
            plt.scatter(x,y, label = 'car'+str(car), alpha = 0.3)
    # nan_indices = np.isnan(np.array(traj_lists['car0x'][i][0:-4])).sum()
    nan_indices = np.isnan(np.array(traj_lists['car0x'][i])).sum()

    
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    if nan_indices > nan_num_prev[0]:
        plt.title(f"CRASH Detected! Suriving Cars:{num_traj-nan_num_prev[0]}")
        nan_num_prev[0] = nan_indices
    else:
        plt.title(f"Position Location distributions over time, Surviving Cars:{num_traj-nan_num_prev[0]}")

    if environment == "roundabout":
        plt.xlim(-30,130)
        plt.ylim(-60,60)
        plt.gca().set_aspect('equal', adjustable='box')
    if environment == "intersection":
        plt.xlim(-80,80)
        plt.ylim(-60,60)
        plt.gca().set_aspect('equal', adjustable='box')
    if environment == "highway":
        plt.xlim(200,500)
        plt.ylim(-10,5)
        
    # plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.grid()

def plot_over_time(traj_lists,num_traj,environment):
    """
    Plot all trajectories over time

    Input:
    traj_lists - All stored trajectories in list
    num_traj - number of trajectoreis
    """

    nan_num_prev =  [0] 

    fig = plt.figure()
    ani = FuncAnimation(fig, 
                        animate, 
                        fargs=(fig,nan_num_prev,traj_lists,num_traj,environment,),
                        frames = 80,
                        repeat = False,
                        interval =100)
    ani.save(filename="trajectory_cars_distribution.gif", writer="ffmpeg")

def find_indices_before_first_nan(arr):
    """
    Finds the index right before the first occurrence of NaN in each column.
    If a column has no NaN, returns the last index.
    
    Parameters:
    arr (numpy.ndarray): 2D numeric array where each row is a list of numbers.
    
    Returns:
    numpy.ndarray: 1D array of indices for each column.
    """
    is_nan = np.isnan(arr)  # Boolean mask of NaNs
    first_nan = np.argmax(is_nan, axis=0)  # Finds first NaN in each column
    all_valid_mask = ~np.any(is_nan, axis=0)  # Columns where no NaNs exist
    first_nan[all_valid_mask] = arr.shape[0]  # Set to last index if no NaNs
    return first_nan - 1  # Get the index before the first NaN

def heatmap_failure_spots(traj_lists):

    arr = traj_lists['car0x']
    # Determine the maximum row length
    max_len = max(len(row) for row in arr)

    # Pad shorter rows with NaNs to make a uniform 2D array
    padded_arr = np.full((len(arr), max_len), np.nan)
    for i, row in enumerate(arr):
        padded_arr[i, :len(row)] = row

    indices = find_indices_before_first_nan(padded_arr)
    
    # Collect all trajectories into one big matrix
    all_x = []
    all_y = []
    
    for traj,index in enumerate(indices):
        if index != -1:
            all_x.append(traj_lists['car0x'][index][traj])
            all_y.append(traj_lists['car0y'][index][traj])

    for i in range(5):
        x = np.array([list[i] for list in traj_lists['car0x']])
        y = np.array([list[i] for list in traj_lists['car0y']])
        all_x.extend(x)
        all_y.extend(y)
    plt.scatter(all_x, all_y, label=f'original trajectory', alpha=0.02, s=3)

    # Use Seaborn's 2D density plot
    sns.kdeplot(x=all_x, 
                y=all_y, 
                fill=True, 
                cmap="Reds", 
                alpha=0.7,
                bw_adjust=0.5,  # Adjust bandwidth to avoid over-smoothing
                thresh=0.1)  # Mask out very low-density regions)

    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Trajectory Density - Failure Analysis")
    plt.grid()
    plt.gca().set_aspect('equal', adjustable='box')

    plt.show()


def bayesian_failure_estimation(failures, total, alpha_prior=1, beta_prior=1):
    """
    Performs Bayesian estimation for failure probability using a Beta-Binomial model.

    Parameters:
    - failures (int): Number of observed failures.
    - total (int): Total number of trials.
    - alpha_prior (float): Prior alpha for Beta distribution (default=1, uninformative prior).
    - beta_prior (float): Prior beta for Beta distribution (default=1, uninformative prior).

    Returns:
    - mean_prob (float): Expected probability of failure.
    - credible_interval (tuple): 95% credible interval for the failure probability.
    """

    # Posterior distribution parameters
    alpha_post = alpha_prior + failures
    beta_post = beta_prior + (total - failures)

    # Compute expected probability (mean of Beta posterior)
    mean_prob = alpha_post / (alpha_post + beta_post)

    # Compute 95% credible interval
    credible_interval = stats.beta.ppf([0.025, 0.975], alpha_post, beta_post)

    # Plot posterior distribution
    x = np.linspace(0, 1, 1000)
    y = stats.beta.pdf(x, alpha_post, beta_post)

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, label=f'Beta({alpha_post}, {beta_post}) Posterior', color='red')
    plt.fill_between(x, y, alpha=0.3, color='red')
    plt.axvline(mean_prob, color='black', linestyle='--', label=f"Mean P(Failure) = {mean_prob:.3f}")
    plt.axvline(credible_interval[0], color='blue', linestyle='dashed', label="95% CI Lower Bound")
    plt.axvline(credible_interval[1], color='blue', linestyle='dashed', label="95% CI Upper Bound")
    plt.xlabel("Failure Probability")
    plt.ylabel("Density")
    plt.title("Bayesian Posterior of Failure Probability")
    plt.legend()
    plt.show()

    return mean_prob, credible_interval

# TODO: Do we need this? Not working well
def pos_heatmap_traj(n_traj,traj_lists):

    # Collect all trajectories into one big matrix
    all_x = []
    all_y = []

    nan_indices = np.isnan( np.array(traj_lists['car0x'][-1][0:-4]))
    for i in range(n_traj):
        x = np.array([list[i] for list in traj_lists['car0x']])
        y = np.array([list[i] for list in traj_lists['car0y']])*-1
        all_x.extend(x)
        all_y.extend(y)
    # plt.scatter(all_x, all_y, label=f'car{i} NaNs', alpha=0.02, s=10)

    # Use Seaborn's 2D density plot
    sns.kdeplot(x=all_x, y=all_y, fill=True, cmap="Reds", alpha=0.2)

    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Trajectory Density - Failure Analysis")
    plt.grid()
    plt.gca().set_aspect('equal', adjustable='box')

    plt.show()


if __name__ == "__main__":

    # Load in Data
    with open('data/data_ppo_roundabout_1000_80.pkl', 'rb') as f: # data_nominal
        nominalTraj = pickle.load(f)


    n_cars = 5
    ind_traj = 1
    n_traj = len(nominalTraj['car0x'][0])
    plot_traj(n_cars, ind_traj, nominalTraj)
    plot_initial(n_cars, nominalTraj)
    heatmap_failure_spots(nominalTraj)

    fail = np.isnan( np.array(nominalTraj['car0x'][-1][0:len(nominalTraj['car0x'][0])])).sum()
    bayesian_failure_estimation(fail, n_traj, alpha_prior=1, beta_prior=1)
    plot_over_time(nominalTraj,n_traj, "roundabout")
    # pos_heatmap_traj(n_traj,nominalTraj)



