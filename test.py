import gym, gym_forestfire
import time

# Usage: pass an action ("N", "S", "E", "W", "D", "") to sim.step(),
# and show the map with sim.render()

sim = gym.make('gym-forestfire-v0')
#cart = gym.make('CartPole-v0')
#lake = gym.make('FrozenLake-v0')
sim.render()
print("Success!")


def run(sim):
    while sim.env.running:
        action = sim.action_space.sample()
        sim.step(action)
        sim.render()
        time.sleep(0.1)


def time_simulation_run():
    import timeit
    setup = """
import gym, gym_forestfire
sim = gym.make('gym-forestfire-v0')
    """
    code = """
sim.reset()
while sim.env.running:
    action = sim.action_space.sample()
    sim.step(action)
    """
    num_runs = 100
    total = timeit.timeit(setup=setup, stmt=code, number=num_runs)
    print("Total:", total, "Average per run", total / num_runs)

