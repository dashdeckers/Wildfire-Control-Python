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

def plot_setyaxis(min, max):
    plt.gca().set_ylim([min, max])
    plt.gca().set_xlim([None, None])

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
        plot_setyaxis(-1500, 2000)
        plot_finish(save_filename)
    else:
        print(f"Warning: No data on total rewards")

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
        plot_add(calc_average_per_k(total_rewards, k), "Averaged reward")
        plot_setyaxis(-1500, 2000)
        plot_finish(save_filename)
    else:
        print(f"Warning: No data on total rewards")

# Plots the running average
def plot_running_average(file, k=100):
    try:
        total_rewards = file[1]['total_rewards']
        save_filename = plots_folder + file[0] + '-(running_average).png'
    except:
        print(f"ERROR: No total_rewards field in log!")
        return
    if total_rewards:
        # Generate the plot
        plot_start(f"Running average reward over time (k = {k})",  \
            "Running average reward", "Episode")
        plot_add(calc_running_average(total_rewards, k), \
            "Running average reward")
        plot_setyaxis(-1500, 2000)
        plot_finish(save_filename)
    else:
        print(f"Warning: No data on total rewards")

# Plot averages per spawning distance (WIP)
def plot_fire_distance(file, k=100):
    # Make sure it doesn't crash on old logs
    try:
        total_rewards = file[1]['total_rewards']
        agent_pos = file[1]['agent_pos']
        metadata = file[1]['metadata']
        save_filename = plots_folder + file[0] + '-(fire_distance).png'
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
        plot_add(calc_running_average(no_dist, k), "1")
        plot_add(calc_running_average(low_dist, k), "2")
        plot_add(calc_running_average(med_dist, k), "3")
        plot_add(calc_running_average(hi_dist, k), "4")
        plot_setyaxis(-1500, 2000)
        plot_finish(save_filename)
    else:
        print(f"Warning: No data on fire distance")


### MATH HELPERS
# Averages some array per some k
def calc_average_per_k(array, k=None):
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
def calc_reduce_array(array, k=100):
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
def calc_running_average(array, k=100):
    return np.convolve(array, np.ones((k,))/k, mode='valid')


### INTERACTIVE THING
def interactive_example():
    from matplotlib.widgets import Slider, Button, RadioButtons

    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)
    t = np.arange(0.0, 1.0, 0.001)
    a0 = 5
    f0 = 3
    delta_f = 5.0
    s = a0 * np.sin(2 * np.pi * f0 * t)
    l, = plt.plot(t, s, lw=2)
    plt.axis([0, 1, -10, 10])

    axcolor = 'lightgoldenrodyellow'
    axfreq = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    axamp = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)

    sfreq = Slider(axfreq, 'Freq', 0.1, 30.0, valinit=f0, valstep=delta_f)
    samp = Slider(axamp, 'Amp', 0.1, 10.0, valinit=a0)


    def update(val):
        amp = samp.val
        freq = sfreq.val
        l.set_ydata(amp*np.sin(2*np.pi*freq*t))
        fig.canvas.draw_idle()


    sfreq.on_changed(update)
    samp.on_changed(update)

    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


    def reset(event):
        sfreq.reset()
        samp.reset()
    button.on_clicked(reset)

    rax = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor=axcolor)
    radio = RadioButtons(rax, ('red', 'blue', 'green'), active=0)


    def colorfunc(label):
        l.set_color(label)
        fig.canvas.draw_idle()
    radio.on_clicked(colorfunc)

    plt.show()


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
            #plot_total_rewards(file)
            #plot_td_error(file)
            #plot_average_reward_per_k(file, 250)
            #plot_running_average(file, 250)
            plot_fire_distance(file, 250)



if __name__ == "__main__":
    interactive_example()
