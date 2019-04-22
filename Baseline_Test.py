import time, gym, gym_forestfire

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import A2C

# this script runs the A2C algorithm on the forestfire environment
# to run this, first execute 'pip install stable-baselines'

# multiprocess environment
n_cpu = 32
env = SubprocVecEnv([lambda: gym.make('gym-forestfire-v0') for i in range(n_cpu)])

model = A2C(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=100000000)

input("Press any key to continue")

def run_optimal(model):
    sim = gym.make('gym-forestfire-v0')
    obs = sim.reset()
    total_reward = 0
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _ = sim.step(action)
        total_reward += reward
        sim.render()
        time.sleep(0.1)
    print("Total reward: ", total_reward)

run_optimal(model)
