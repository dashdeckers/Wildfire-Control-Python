import json, pprint

pp  = pprint.PrettyPrinter()
pp1 = pprint.PrettyPrinter(depth=1)
pp2 = pprint.PrettyPrinter(depth=2)
pp3 = pprint.PrettyPrinter(depth=3)

with open("Logs/Time:May-14-20:43:23", 'r') as file:
    logs = json.load(file)


pp.pprint(logs)


def show_all(self):
    self.play_optimal()
    self.show_rewards()
    self.show_td_error()
    self.show_decay()

# Gives an impression on how the epsilon decays over time
def show_decay(self, amount_of_values=5):
    n_episodes = len(self.logs['epsilons'])
    k = int(n_episodes / amount_of_values)

    print(f"Epsilon starts at", round(self.logs['epsilons'][0], 3), \
                "and ends at", round(self.logs['epsilons'][n_episodes-1], 3))

    for i, eps in enumerate(self.logs['epsilons']):
        if i % k == 0:
            print(f"\tEpisode {i+1}: ", round(eps, 3))

# Plot the cumulative rewards over time
def show_rewards(self):
    plt.plot(self.logs['total_rewards'])
    plt.title("Cumulative reward over time")
    plt.ylabel("Reward Values")
    plt.xlabel("Episodes")
    plt.show()

# Plot the TD errors over time
def show_td_error(self):
    # Each TD error entry in logs is a list of TD errors of length 
    # batch_size for each loop iteration in replay()
    tderrors = list()
    for group in self.logs['TD_errors']:
        tderrors.append(sum(group) / len(group))
    plt.plot(tderrors)
    plt.title("TD errors over time")
    plt.ylabel("TD Errors")
    plt.xlabel("Episodes")
    plt.show()

# Prints or plots the average cumulative reward per k episodes
def average_reward_per_k_episodes(self, k, plot=False):
    # Calculate the average cumulative reward
    n_episodes = len(self.logs['total_rewards'])
    rewards_per_k = np.split(np.array(self.logs['total_rewards']), n_episodes/k)
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
