import gym
import gym_forestfire
import time

# Usage: pass an action ("N", "S", "E", "W", "Dig") to step()
# and show the map with env.render()

env = gym.make('gym-forestfire-v0')
env2 = gym.make('CartPole-v0')
env.render()
print("Success!")


def run(env):
    while env.env.running:
        env.step("")
        env.render()
        time.sleep(0.1)
