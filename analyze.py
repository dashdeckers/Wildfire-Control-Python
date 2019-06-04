import os, json, pprint
import matplotlib.pyplot as plt
import numpy as np

# Global settings/variables
logs_folder = "Logs/"
plots_folder = "Plots/"
smoothing_factor = 0.99

'''
Logs data structure:

Key:
    Type
    Description
    Size

{
    TD_errors:
        List of lists (should be list of lists of lists though)
        The difference between every state / prediction in replay()
        Batch size * total number of states

    total_rewards:
        List
        The total cumulative reward for every episode
        Total number of episodes

    agent_pos:
        List
        The agent starting positions for every episode
        Total number of episodes

    maps:
        List of lists
        Each entry contains the episode number and a string of the final state
        Only every printed map is saved

    best_reward:
        Float
        The highscore reward across all episodes

    init_memories:
        Int
        Number of memories the agent was initialized with before learning

    n_episodes:
        Int
        Total number of episodes the algorithm learnt for

    total_time:
        Int
        Total number of seconds the run took to run

    metadata:
        Dictionary
        All the DQN parameters and simulation constants
}
'''

### FILE LOADING
# Make sure the log and plot folders exist
def verify_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Directory " + folder_name + " missing, empty folder created!")

# Make a list of filenames of everything in logs_folder
def get_log_filenames():
    all_filenames = list()
    print(f"The following log files are available:")
    for root, dirs, files in os.walk(logs_folder + "."):
        for filename in files:
            all_filenames.append(filename)
    return all_filenames

# Let user select a filename given all the filenames
def select_file(all_filenames):
    for idx, filename in enumerate(all_filenames):
        print(f"\t[{idx}] {filename}")
    selection = input(f"Select one [0-{idx}]: ")
    print("")
    return all_filenames[int(selection)]

# Load a file given its name
def load_file(filename):
    with open(logs_folder + filename) as file:
        return (filename, json.load(file))

# Print relevant constants given the loaded file
def print_constants(file):
    if file[1]['metadata']:
        print(f"The selected log contains the following constants:")
        metadata = file[1]['metadata']
        print(f"\tEpisodes:\t{file[1]['n_episodes']}", end="")
        print(f" ({round(file[1]['total_time']/3600, 1)}hrs)")
        print(f"\tMap size:\t{max(metadata['width'], metadata['height'])}")
        print(f"\tMemories:\t{file[1]['init_memories']}")
        print(f"\tDebug level:\t{metadata['debug']}")
        print(f"\tDecay:\t{metadata['eps_decay_rate']}")
        print(f"\tGamma:\t{metadata['gamma']}")
        print(f"\tAlpha:\t{metadata['alpha']}")
    else:
        print(f"ERROR: No metadata found!")
    print("")


### PLOT MAKING/SHOWING/SAVING
# Clear plot and start a new one
def plot_start(title, ylabel, xlabel):
    plt.clf()
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)

# Add data with respective name to the current plot
def plot_add(data, given_label):
    plt.plot(data, label=given_label)

# Define maximum/minimum y values
def plot_setyaxis(min, max):
    plt.gca().set_ylim([min, max])
    plt.gca().set_xlim([None, None])

# Add a horizontal line indicating max value
def plot_maxline(array):
    maximum = max(array)
    plt.plot([0, len(array)], [maximum, maximum])
    plt.text(0, maximum, f"{round(maximum)}")

# Add a horizontal line indicating avg value
def plot_avgline(array):
    import statistics
    avg = statistics.mean(array)
    plt.plot([0, len(array)], [avg, avg])
    plt.text(0, avg, f"{round(avg)}")

# Save and show the final plot
def plot_finish(save_filename):
    plt.legend()
    plt.savefig(save_filename)
    print(f"Generated {save_filename}")
    plt.show()


### PLOT DEFINITIONS
# Plot the total rewards over time
def plot_total_rewards(file):
    try:
        total_rewards = file[1]['total_rewards']
        verify_folder(plots_folder + file[0])
        save_filename = plots_folder + file[0] + '/total_rewards.png'
    except:
        print(f"ERROR: No total_rewards field in log!")
        return
    if total_rewards:
        # Generate the plot
        plot_start("Total reward over time", "Total reward", "Episode")
        plot_add(calc_smooth(total_rewards), "Total reward")
        plot_setyaxis(-1500, 2000)
        plot_maxline(calc_smooth(total_rewards))
        plot_avgline(calc_smooth(total_rewards))
        plot_finish(save_filename)
    else:
        print(f"Warning: No data on total rewards")

# Plot the TD errors over time
def plot_td_error(file):
    try: 
        TD_errors = file[1]['TD_errors']
        verify_folder(plots_folder + file[0])
        save_filename = plots_folder + file[0] + '/td_errors.png'
    except:
        print(f"ERROR: No TD_error field in log!")
        return
    if TD_errors:
        # Generate the plot
        plot_start("TD-error over time", "TD-error", "Episode")
        plot_add(TD_errors, "TD-error")
        plot_finish(save_filename)
    else:
        print(f"Warning: No data on TD-error")

# Plots the average total reward per k episodes
def plot_average_reward_per_k(file, k = None):
    try:
        total_rewards = file[1]['total_rewards']
        verify_folder(plots_folder + file[0])
        save_filename = plots_folder + file[0] + '/average_rewards.png'
    except:
        print(f"ERROR: No total_rewards field in log!")
        return
    if total_rewards:
    # Generate the plot
        avg_per_k = calc_average_per_k(total_rewards, k)
        if not avg_per_k:
            return
        plot_start(f"Average reward over time (k = {k})", \
                "Average reward", "Episode * k")
        plot_add(calc_smooth(avg_per_k), "Averaged reward")
        plot_setyaxis(-1500, 2000)
        plot_finish(save_filename)
    else:
        print(f"Warning: No data on total rewards")

# Plot averages per spawning distance (WIP)
def plot_fire_distance(file):
    # Make sure it doesn't crash on old logs
    try:
        total_rewards = file[1]['total_rewards']
        agent_pos = file[1]['agent_pos']
        metadata = file[1]['metadata']
        verify_folder(plots_folder + file[0])
        save_filename = plots_folder + file[0] + '/fire_distance.png'
    except:
        print(f"ERROR: Missing datafields for fire distance in log!")
        return
    # Continue if shit's okay
    if total_rewards and agent_pos and metadata:
        # Initialize all required datafields
        fire_x, fire_y = (int(metadata['width']/2), \
                        int(metadata['height']/2))
        no_dist, low_dist, med_dist, hi_dist = ([] for i in range(4))
        # Split rewards according to Manhattan distance
        for reward, agent_pos in zip(total_rewards, agent_pos):
            agent_x, agent_y = agent_pos
            from scipy.spatial import distance
            dist = distance.cityblock([fire_x, fire_y], [agent_x, agent_y])
            if dist == 1:
                no_dist.append(reward)
            elif dist == 2:
                low_dist.append(reward)
            elif dist == 3:
                med_dist.append(reward)
            elif dist == 4:
                hi_dist.append(reward)
        # Generate the plot
        plot_start(f"Total reward/episode per Manhattan distance to the fire", \
            "Total reward", "Episode")
        if not (no_dist and low_dist and med_dist and hi_dist):
            print("ERROR: Not all distances to the fire are represented")
            return
        plot_add(calc_smooth(no_dist), "1")
        plot_add(calc_smooth(low_dist), "2")
        plot_add(calc_smooth(med_dist), "3")
        plot_add(calc_smooth(hi_dist), "4")
        plot_setyaxis(-1500, 2000)
        plot_finish(save_filename)
    else:
        print(f"Warning: No data on fire distance")

# Plots the reward gained compared to last k values
def plot_reward_gained(file, k = 1000):
    try:
        total_rewards = file[1]['total_rewards']
        verify_folder(plots_folder + file[0])
        save_filename = plots_folder + file[0] + '/reward_gained.png'
    except:
        print(f"ERROR: No total_rewards field in log!")
        return
    if total_rewards:
        reward_gains = []
        avg_so_far = total_rewards[0]

        for idx, reward in enumerate(total_rewards):
            start = 0
            if idx - k > 0:
                start = idx - k

            if len(total_rewards[start:idx:]) == 0:
                reward_gains.append(total_rewards[1]-total_rewards[0])
            else:
                div_len = len(total_rewards[start:idx:])
                avg_so_far = sum(total_rewards[start:idx:]) / div_len
                reward_gains.append(reward - avg_so_far)

        # Generate the plot
        plot_start(f"Reward gained over time", \
            "Reward gained", "Episode")
        plot_add(calc_smooth(reward_gains), "Reward gained")
        plot_setyaxis(-1500, 2000)
        plot_maxline(calc_smooth(reward_gains))
        plot_avgline(calc_smooth(reward_gains))
        plot_finish(save_filename)
    else:
        print(f"Warning: No data on total rewards")


### MATH HELPERS
# Averages some array per some k
def calc_average_per_k(array, k = None):
    length = len(array)
    if not length % k == 0:
        print("ERROR: k has to be a divisor of number of episodes "
              "to calculate the average reward per k")
        return
    # If k was not given, find a decent factor of length to use
    if k is None:
        k = 1
        for i in range(3, int(length / 2)):
            if length % i == 0:
                k = i
                break
    # Calculate the average total reward
    rewards_per_k = np.split(np.array(array), length/k)
    avg_reward_per_k = list()
    for group in rewards_per_k:
        avg_reward_per_k.append(sum(group) / k)
    return avg_reward_per_k

# Takes as many k-size averages as possible from an array,
# discards the remainder though!
def calc_reduce_array(array, k = 100):
    reduced, temp = ([] for i in range(2))
    idx = 0
    while len(array)- 1 - idx >= k:
        temp.append(array[idx])
        if not len(temp) % k:
            reduced.append(int(sum(temp)/k))
            temp = []
        idx += 1
    return reduced

# Calculate the running average of an array given k
def calc_running_average(array, k = 100):
    return np.convolve(array, np.ones((k,))/k, mode='valid')

# Smooths the dataset given a smoothing factor
def calc_smooth(array, weight = -1):
    if weight == -1:
        weight = smoothing_factor
    last = array[0]
    smoothed = list()
    for value in array:
        smoothed_val = last * weight + (1 - weight) * value
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

### MAIN
def main():
    # Make sure the necessary folders exist
    verify_folder(logs_folder)
    verify_folder(plots_folder)

    correct_file = "n"
    while correct_file == "n":
        # Lets the user view files and select one
        all_filenames = get_log_filenames()
        selected_filename = select_file(all_filenames)
        file = load_file(selected_filename)

        # Print relevant constant from the loaded file
        print_constants(file)

        # Make the file contents available globally
        # Usage in CLI:     print(log['total_rewards'])
        global log
        log = file[1]

        correct_file = input("Make plots? (y/n/c): ")
        print("")
        if correct_file == "c":
            print(f"Canceling file selection..")
            break
        if correct_file == "n":
            print(f"Reselecting file..")
        else:
            # Defines the to-be generated plots using the selected logfile
            plot_total_rewards(file)
            plot_td_error(file)
            plot_average_reward_per_k(file, 100)
            plot_fire_distance(file)
            plot_reward_gained(file)

if __name__ == "__main__":
	main()
