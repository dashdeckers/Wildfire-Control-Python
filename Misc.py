import time

# to be able to use getch() to get a character and not wait for enter
try:
    # Win32
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

# acts based on keyboard input
def run_human(sim):
    done = False
    sim.reset()
    while not done:
        print("WASD to move, Space to dig, 'n' to wait, 'q' to quit.\n")
        char = getch()
        if char == 'q':
            break
        elif char == 'w':
            _, _, done, _ = sim.step("N")
        elif char == 's':
            _, _, done, _ = sim.step("S")
        elif char == 'd':
            _, _, done, _ = sim.step("E")
        elif char == 'a':
            _, _, done, _ = sim.step("W")
        elif char == ' ':
            _, _, done, _ = sim.step("D")
        elif char == 'n':
            _, _, done, _ = sim.step(" ")
        sim.render()

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
