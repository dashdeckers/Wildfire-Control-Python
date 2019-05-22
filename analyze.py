import os, json, pprint
import matplotlib.pyplot as plt
import numpy as np

ppx = pprint.PrettyPrinter()
pp1 = pprint.PrettyPrinter(depth=1)
pp2 = pprint.PrettyPrinter(depth=2)
pp3 = pprint.PrettyPrinter(depth=3)

# Global settings/variables
logs_folder = "Logs/"
plots_folder = "Plots/"
log = None

'''
TODO: Plots need to show the constants. Main() is nice, but real data presentation
should probably be done interactively because we want to show differences between
changes to a specific constant and not all constants (although if we only vary some
of the parameters, we could always show those in the legend)

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

    all_rewards:
        List of lists
        The reward collected in every visited state of every episode
        Total number of states

    total_rewards:
        List
        The total cumulative reward for every episode
        Total number of episodes

    epsilons:
        List
        The epsilon value for every episode
        Total number of episodes

    maps:
        List of lists
        Each entry contains the episode number and a string of the final state
        Only every printed map is saved

    best_reward:
        Float
        The highscore reward across all episodes

    deaths:
        Int
        Number of times the agent has died across all episodes

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

# Save and show the final plot
def plot_finish(save_filename):
    plt.legend()
    plt.show()
    plt.savefig(save_filename)
    print(f"Generated {save_filename}")


### PLOT DEFINITIONS
# Plot the cumulative rewards over time
def plot_total_rewards(file):
    try:
        total_rewards = file[1]['total_rewards']
        save_filename = plots_folder + file[0] + '-(total_rewards).png'
    except:
        print(f"ERROR: No total_rewards field in log!")
        return
    if total_rewards:
        # Generate the plot
        plot_start("Cumulative reward over time", "Total reward", "Episode")
        plot_add(total_rewards, "Total reward")
        plot_finish(save_filename)
    else:
        print(f"Warning: No data on total rewards")

# Gives an impression on how the epsilon decays over time
def plot_decay(file):
    try:
        epsilons = file[1]['epsilons']
        save_filename = plots_folder + file[0] + '-(epsilons).png'
    except:
        print(f"ERROR: No epsilons field in log!")
        return
    if epsilons:
        # Generate the plot
        plot_start("Epsilon decay over time", "Epsilon", "Episode")
        plot_add(epsilons, "Epsilon")
        plot_finish(save_filename)
    else:
        print(f"Warning: No data on epsilon decay")

# Plot the TD errors over time
def plot_td_error(file):
    try: 
        TD_errors = file[1]['TD_errors']
        save_filename = plots_folder + file[0] + '-(td_errors).png'
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

# Plots the average cumulative reward per k episodes
def plot_average_reward_per_k(file, k=None):
    try:
        total_rewards = file[1]['total_rewards']
        save_filename = plots_folder + file[0] + '-(average_rewards).png'
    except:
        print(f"ERROR: No total_rewards field in log!")
        return
    if total_rewards:
    # Generate the plot
        plot_start(f"Average reward over time (k = {k})", \
                "Average reward", "Episode/k")
        plot_add(average_per_k(total_rewards, k), "Averaged reward")
        plot_finish(save_filename)
    else:
        print(f"Warning: No data on total rewards")

# Plot average reward per quadrant
def plot_quadrant_reward(file, k=None):
    # Make sure it doesn't crash on old logs
    try:
        total_rewards = file[1]['total_rewards']
        agent_pos = file[1]['agent_pos']
        metadata = file[1]['metadata']
        save_filename = plots_folder + file[0] + '-(quadrant_reward).png'
    except:
        print(f"ERROR: Missing datafields for quadrant rewards in log!")
        return
    # Continue if shit's okay
    if total_rewards and agent_pos and metadata:
        # Initialize all required datafields
        halfwidth = int(metadata['width']/2)
        halfheight = int(metadata['height']/2)
        topleft, topright, bottomleft, bottomright = \
                ([] for i in range(4))
        # Assign the total reward of each position to
        # the relevant quadrant
        for idx, pos in enumerate(agent_pos):
            x, y = pos
            # Case where x < 5 and y < 5
            if x < halfwidth and y < halfheight:
                topleft.append(int(total_rewards[idx]))
            # Case where x < 5 and y >= 5
            elif x < halfwidth and y >= halfheight:
                bottomleft.append(int(total_rewards[idx]))
            # Case where x >= 5 and y < 5
            elif x >= halfwidth and y < halfheight:
                topright.append(int(total_rewards[idx]))
            # Case where x >= 5 and y >= 5
            elif x >= halfwidth and y >= halfheight:
                bottomright.append(int(total_rewards[idx]))
        # Generate the plot
        plot_start(f"Total reward over time per quadrant (k = {k})", \
            "Total reward", "Episode/k")
        plot_add(reduce_array(topleft, k), "Top left")
        plot_add(reduce_array(bottomleft, k), "Bottom right")
        plot_add(reduce_array(topright, k), "Top right")
        plot_add(reduce_array(bottomright, k), "Bottom right")
        plot_finish(save_filename)
    else:
        print(f"Warning: No data on quadrant rewards")


### MATH HELPERS
# Averages some array per some k
def average_per_k(array, k=None):
    length = len(array)
    # If k was not given, find a decent factor of length to use
    if k is None:
        k = 1
        for i in range(3, int(length / 2)):
            if length % i == 0:
                k = i
                break
    # Calculate the average cumulative reward
    rewards_per_k = np.split(np.array(array), length/k)
    avg_reward_per_k = list()
    for group in rewards_per_k:
        avg_reward_per_k.append(sum(group) / k)
    return avg_reward_per_k

# Takes as many k-size averages as possible from an array,
# discards the remainder though!
def reduce_array(array, k=100):
    reduced, temp = ([] for i in range(2))
    idx = 0
    while len(array)- 1 - idx >= k:
        temp.append(array[idx])
        if not len(temp) % k:
            reduced.append(int(sum(temp)/k))
            temp = []
        idx += 1
    return reduced
        

### MAIN
def main():
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
            plot_decay(file)
            plot_td_error(file)
            plot_average_reward_per_k(file, 100)
            plot_quadrant_reward(file, 100)

if __name__ == "__main__":
    main()
