import os, json, pprint
import matplotlib.pyplot as plt
import numpy as np

ppx = pprint.PrettyPrinter()
pp1 = pprint.PrettyPrinter(depth=1)
pp2 = pprint.PrettyPrinter(depth=2)
pp3 = pprint.PrettyPrinter(depth=3)

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

def get_log_files():
    all_logs = list()
    logs_location = "Logs/"
    for filename in os.listdir(logs_location):
        with open(logs_location + filename, 'r') as file:
            all_logs.append((filename, json.load(file)))
    return all_logs

def show_all(logs):
    show_rewards(logs)
    show_td_error(logs)
    show_decay(logs)
    show_average_reward_per_k_episodes(logs)

# Gives an impression on how the epsilon decays over time
def show_decay(logs, amount_of_values=5):
    n_episodes = len(logs['epsilons'])
    if not n_episodes:
        print("No data on epsilon decay...")
        return

    k = int(n_episodes / amount_of_values)

    print(f"Epsilon starts at", round(logs['epsilons'][0], 3), \
                 "and ends at", round(logs['epsilons'][n_episodes-1], 3))

    for i, eps in enumerate(logs['epsilons']):
        if i % k == 0:
            print(f"\tEpisode {i+1}: ", round(eps, 3))

# Plot the cumulative rewards over time
def show_rewards(logs):
    if not len(logs['total_rewards']):
        print("No data on total rewards...")
        return
    plt.plot(logs['total_rewards'])
    plt.title("Cumulative reward over time")
    plt.ylabel("Reward Values")
    plt.xlabel("Episodes")
    plt.show()

# Plot the TD errors over time
def show_td_error(logs):
    if not len(logs['TD_errors']):
        print("No data on TD errors...")
        return
    # Each TD error entry in logs is a list of TD errors of length 
    # batch_size for each loop iteration in replay()
    tderrors = list()
    for group in logs['TD_errors']:
        tderrors.append(sum(group) / len(group))
    plt.plot(tderrors)
    plt.title("TD errors over time")
    plt.ylabel("TD Errors")
    plt.xlabel("Episodes")
    plt.show()

# Prints or plots the average cumulative reward per k episodes
def show_average_reward_per_k_episodes(logs, k=None, plot=True):
    n_episodes = len(logs['total_rewards'])
    if not n_episodes:
        print("No data on total rewards...")
        return

    # If k was not given, find a decent factor of n_episodes to use
    if k is None:
        k = 1
        for i in range(3, int(n_episodes / 2)):
            if n_episodes % i == 0:
                k = i
                break

    # Calculate the average cumulative reward
    rewards_per_k = np.split(np.array(logs['total_rewards']), n_episodes/k)
    avg_reward_per_k = list()
    for group in rewards_per_k:
        avg_reward_per_k.append(sum(group) / k)

    # Plot or print the result
    if plot:
        plt.plot(avg_reward_per_k)
        plt.title(f"Average reward per {k} episodes")
        plt.show()
    else:
        count = k
        print(f"Average reward per {k} episodes:")
        for reward in avg_reward_per_k:
            print(count, ":", reward)
            count += k

def main():
    all_logs = get_log_files()

    max_count = 3
    for filename, logs in all_logs:
        max_count -= 1
        if max_count == 0:
            break

        print(f"\n------- Showing logfile from: {filename}")
        show_all(logs)

if __name__ == "__main__":
    main()
