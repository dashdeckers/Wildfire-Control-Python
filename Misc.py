import time
from keras import backend as K
from keras.callbacks import TensorBoard

# custom tensorboard class which also records TD_error, reward and learning rate
class Custom_TensorBoard(TensorBoard):
    def __init__(self, log_dir, histogram_freq=0, batch_size=1,
                 write_graph=True, write_grads=True):
        super().__init__(log_dir=log_dir, histogram_freq=histogram_freq,
                         batch_size=batch_size, write_graph=write_graph,
                         write_grads=write_grads)

    def on_epoch_end(self, epoch, logs, TD_error, reward):
        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        logs.update({'td_error': TD_error})
        logs.update({'reward': reward})
        super().on_epoch_end(epoch, logs)

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
    key_map = {'w':'N', 's':'S', 'd':'E', 'a':'W', ' ':'D', 'n':' '}
    done = False
    total_reward = 0
    sim.reset()
    sim.render()
    while not done:
        print("WASD to move, Space to dig, 'n' to wait, 'q' to quit.")
        print("'i' to inspect a single cell, 'p' to print general info.\n")
        char = getch()
        if char == 'q':
            break
        elif char in key_map:
            _, reward, done, _ = sim.step(key_map[char])
            total_reward += reward
        elif char == 'i':
            print("Inspect a cell")
            x = int(input("X coordinate: "))
            y = int(input("Y coordinate: "))
            sim.W.inspect((x, y))
        elif char == 'p':
            print("General Info")
            sim.W.print_info()
        sim.render()
    print(f"Total Reward: {total_reward}")

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
sim.step("D")
while sim.W.RUNNING:
    sim.step(" ")
    """
    total = timeit.timeit(setup=setup, stmt=code, number=num_runs)
    r1, r2 = round(total, 4), round(total / num_runs, 4)
    print("Total:", r1, "Average per run", r2)

