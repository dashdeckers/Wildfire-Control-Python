import gym, gym_forestfire
import time

# Usage: pass an action ("N", "S", "E", "W", "D", "") to step()
# and show the map with env.render()

env = gym.make('gym-forestfire-v0')
env.render()
print("Success!")


def run(env):
    while env.env.running:
        action = env.action_space.sample()
        env.step(action)
        env.render()
        time.sleep(0.1)

def time_simulation_run():
    import timeit
    setup = """
import gym, gym_forestfire
env = gym.make('gym-forestfire-v0')
    """
    code = """
env.reset()
while env.env.running:
    action = env.action_space.sample()
    env.step(action)
    """
    num_runs = 100
    total = timeit.timeit(setup=setup, stmt=code, number=num_runs)
    print("Total:", total, "Average per run", total / num_runs)

