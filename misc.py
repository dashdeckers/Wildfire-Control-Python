import numpy as np
import time

# Makes a function getch() which gets a char from user without waiting for enter
try:
    # Windows
    from msvcrt import getch
except ImportError:
    # UNIX
    def getch():
        import sys, tty, termios
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            return sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

# Runs the simulation based on keyboard input / human control
def run_human(sim, DQN=None):
    key_map = {'w':0, 's':1, 'd':2, 'a':3, ' ':4, 'n':5}
    done = False
    total_reward = 0
    state = sim.reset()
    state = np.reshape(state, [1] + list(state.shape))
    sim.render()
    while not done:
        print("WASD to move, Space to dig,")
        print("'n' to wait, 'q' to quit.")
        print("'v' to show Q-values,")
        print("'i' to inspect a single cell,")
        print("'l' to show a layer of the map,")
        print("'p' to print general info,")
        print("'m' to print all metadata info.\n")
        char = getch()
        if char == 'q':
            return "Cancelled"
        elif DQN is not None and len(DQN.sim.W.border_points) == 0:
            done = True
        elif char in key_map:
            action = key_map[char]
            # Do action, observe environment
            sprime, reward, done, _ = sim.step(action)
            sprime = np.reshape(sprime, [1] + list(sprime.shape))
            # Store experience in memory
            if DQN is not None:
                DQN.remember(state, action, reward, sprime, done)
            # Current state is now next state
            state = sprime
            total_reward += reward
        elif char == 'v':
            if DQN is not None:
                DQN.show_best_action()
            else:
                print("Only works if you pass a DQN object to this function")
        elif char == 'i':
            print("Inspect a cell")
            x = int(input("X coordinate: "))
            y = int(input("Y coordinate: "))
            sim.W.inspect((x, y))
        elif char == 'l':
            inp = input("Which layer? (string input) ")
            if inp in sim.layer:
                sim.W.show_layer(inp)
            else:
                sim.W.show_layer()
        elif char == 'p':
            print("General Info")
            sim.W.print_info(total_reward)
        elif char == 'm':
            sim.W.print_metadata()
        else:
            print("Invalid action, not good if collecting memories for DQN.\n")
        sim.render()
    print(f"Total Reward: {total_reward}")

# Time the simulation speed
def time_simulation_run(num_runs=100):
    import timeit
    setup = """
from Simulation.forest_fire import ForestFire
sim = ForestFire()
    """
    code = """
sim.reset()
sim.step("D")
while sim.W.RUNNING:
    sim.step(" ")
    """
    total = timeit.timeit(setup=setup, stmt=code, number=num_runs)
    r1, r2 = round(total, 4), round(total / num_runs, 4)
    print("Total:", r1, "Average per run", r2)

