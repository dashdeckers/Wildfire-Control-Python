import os, json, pprint
import matplotlib.pyplot as plt
import numpy as np

# Global settings/variables
logs_folder = "Logs/Process/"
plots_folder = "Plots/"
smoothing_factor = 0.99


### FILE LOADING
# Make sure the log and plot folders exist
def verify_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Directory " + folder_name + " missing, empty folder created!")

# Make a list of filenames of everything in logs_folder
def get_log_filenames():
    all_filenames = list()
    for root, dirs, files in os.walk(logs_folder + "."):
        for filename in files:
            all_filenames.append(filename)
    return sorted(all_filenames)

# Let user select a filename given all the filenames
def select_file(all_filenames):
    print(f"The following log files are available:")
    for idx, filename in enumerate(all_filenames):
        print(f"\t[{idx}]\t{filename}")
    selection = input(f"Select one [0-{idx}]: ")
    label = input(f"Name the logfile: ")
    print("")
    return [all_filenames[int(selection)], label]

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
def plot_total_rewards(selected_files):
    # Start the plot
    folder_name = str()
    plot_start("Total reward over time", "Total reward", "Episode")
    # Populate plot
    for selected_file in selected_files:
        file = load_file(selected_file[0])
        total_rewards = file[1]['total_rewards']
        plot_add(calc_smooth(total_rewards), selected_file[1])
        folder_name = folder_name + selected_file[1]
    # Finish plot
    plot_setyaxis(-1500, 2000)
    verify_folder(plots_folder + folder_name)
    save_filename = plots_folder + folder_name + '/total_rewards.png'
    plot_finish(save_filename)

# Plots the average total reward per k episodes
def plot_average_reward_per_k(selected_files, k = None):
    # Start plot
    folder_name = str()
    plot_start(f"Average reward over time (k = {k})", \
                "Average reward", "Episode * k")
    # Populate plot
    for selected_file in selected_files:
        file = load_file(selected_file[0])
        total_rewards = file[1]['total_rewards']
        try:
            avg_per_k = calc_average_per_k(total_rewards, k)
        except:
            avg_per_k = calc_reduce_array(total_rewards, k)
        plot_add(calc_smooth(avg_per_k), selected_file[1])
        folder_name = folder_name + selected_file[1]
    #Finish plot
    plot_setyaxis(-1500, 2000)
    verify_folder(plots_folder + folder_name)
    save_filename = plots_folder + folder_name + '/average_rewards.png'
    plot_finish(save_filename)

# Plot the percent of agent deaths per k runs
def plot_agent_deaths(selected_files, k=100):
    # Start plot
    folder_name = str()
    plot_start(f"Agent deaths per {k} epsiodes",
            "Percent dead", f"Episode * {k}")
    # Populate plot
    for selected_file in selected_files:
        file = load_file(selected_file[0])
        agent_deaths = file[1]['agent_deaths']
        n_episodes = file[1]['n_episodes']

        if n_episodes % k != 0:
            print("k is not a divisor of n_episodes!")
            return
        avgs = list()
        ticks = list()
        for i in range(int(n_episodes/k)):
            avgs.append(sum(agent_deaths[i * k : (i+1) * k]) / k)
            ticks.append(i)

        plot_add(avgs, selected_file[1])
        folder_name = folder_name + selected_file[1]
    # Finish plot
    verify_folder(plots_folder + folder_name)
    save_filename = plots_folder + folder_name + '/agent_deaths.png'
    plot_finish(save_filename)


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
    answer = input(f"Interactive or hardcoded? [i/h]: ")
    if answer == "i":
        selecting_files = True
        selected_files = []
        all_filenames = get_log_filenames()

        while selecting_files:
            selected_file = select_file(all_filenames)
            selected_files.append(selected_file)
            all_filenames.remove(selected_file[0])

            if len(selected_files) >= 2:
                file_select_answer = input(f"Select more files? [y/n]: ")
                if file_select_answer == "n":
                    selecting_files = False
        print("")

        plot_total_rewards(selected_files)
        plot_agent_deaths(selected_files)
    else:
        # Define how logfiles should be grouped
        all_filenames = get_log_filenames()
        first_files, second_files, third_files = ([] for i in range(3))
        for filename in all_filenames:
            if "DDQN" in filename:
                first_files.append(filename)
                continue
            if "DQN" in filename:
                second_files.append(filename)
                continue
            if "SARSA" in filename:
                third_files.append(filename)
                continue

        # Check if all file groups are equal to 10 (runs)
        if len(first_files) == 10 and \
            len(first_files) == len(second_files) and \
            len(second_files) == len(third_files):
            print("\tSanity check OK!")
        else:
            print("\tSanity check FAIL! Plot not reliable.")
            exit()

        # Average all arrays per group
        first_avg, second_avg, third_avg = ([] for i in range(3))
        temp = []
        for filename in first_files:
            file = load_file(filename)
            temp.append(file[1]['total_rewards'])
            first_avg = np.average(np.array(temp), axis=0)
        temp = []
        for filename in second_files:
            file = load_file(filename)
            temp.append(file[1]['total_rewards'])
            second_avg = np.average(np.array(temp), axis=0)
        temp = []
        for filename in third_files:
            file = load_file(filename)
            temp.append(file[1]['total_rewards'])
            third_avg = np.average(np.array(temp), axis=0)

        # Start the plot
        folder_name = "HARDCODED"
        plot_start("Total reward over time", "Total reward", "Episode")
        # Populate plot
        plot_add(calc_smooth(first_avg), "DDQN")
        plot_add(calc_smooth(second_avg), "DQN")
        plot_add(calc_smooth(third_avg), "SARSA")
        # Finish plot
        plot_setyaxis(-1500, 2000)
        verify_folder(plots_folder + folder_name)
        save_filename = plots_folder + folder_name + '/total_rewards_test.png'
        plot_finish(save_filename)

if __name__ == "__main__":
    main()
