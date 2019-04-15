import time

# acts randomly
def run_random(sim):
    done = False
    sim.reset()
    while not done:
        action = sim.action_space.sample()
        _, _, done, _ = sim.step(action)
        sim.render()
        time.sleep(0.1)

# times the simulation
def time_simulation_run(num_runs=100):
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
    total = timeit.timeit(setup=setup, stmt=code, number=num_runs)
    print("Total:", total, "Average per run", total / num_runs)
