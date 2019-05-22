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
    if file[1]['total_rewards']:
        total_rewards = file[1]['total_rewards']
        save_filename = plots_folder + file[0] + '-(total_rewards).png'
        plot_start("Cumulative reward over time", "Total reward", "Episode")
        plot_add(total_rewards, "Total reward")
        plot_finish(save_filename)
    else:
        print(f"Warning: No data on total rewards...")

# Plots the average cumulative reward per k episodes
def plot_average_reward_per_k(file, k=None):
    if file[1]['total_rewards']:
        total_rewards = file[1]['total_rewards']
        n_episodes = len(total_rewards)
        # If k was not given, find a decent factor of n_episodes to use
        if k is None:
            k = 1
            for i in range(3, int(n_episodes / 2)):
                if n_episodes % i == 0:
                    k = i
                    break
        # Calculate the average cumulative reward
        rewards_per_k = np.split(np.array(total_rewards), n_episodes/k)
        avg_reward_per_k = list()
        for group in rewards_per_k:
            avg_reward_per_k.append(sum(group) / k)
        # Generate the plot
        save_filename = plots_folder + file[0] + '-(average_rewards).png'
        plot_start(f"Average reward over time (k = {k})", \
                "Average reward", "Episode/k")
        plot_add(avg_reward_per_k, "Averaged reward")
        plot_finish(save_filename)
    else:
        print(f"Warning: No data on total rewards...")

# Gives an impression on how the epsilon decays over time
def plot_decay(file):
    if file[1]['epsilons']:
        epsilons = file[1]['epsilons']
        save_filename = plots_folder + file[0] + '-(epsilons).png'
        plot_start("Epsilon decay over time", "Epsilon", "Episode")
        plot_add(epsilons, "Epsilon")
        plot_finish(save_filename)
    else:
        print(f"Warning: No data on epsilon decay...")

# Plot the TD errors over time
def plot_td_error(file):
    if file[1]['TD_errors']:
        TD_errors = file[1]['TD_errors']
        save_filename = plots_folder + file[0] + '-(td_errors).png'
        plot_start("TD-error over time", "TD-error", "Episode")
        plot_add(TD_errors, "TD-error")
        plot_finish(save_filename)
    else:
        print(f"Warning: No data on TD-error...")


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

        correct_file = input("Make plots? (y/n): ")
        if correct_file == "n":
            print(f"Reselecting file..")
        else:
            # Defines the to-be generated plots using the selected logfile
            plot_total_rewards(file)
            plot_decay(file)
            plot_td_error(file)
            plot_average_reward_per_k(file, 100)
        print("")

if __name__ == "__main__":
    main()
