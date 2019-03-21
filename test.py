import gym
import gym_forestfire

# For some reason, env.step() prints a load of crap but it still works.
# Usage: pass an action ("N", "S", "E", "W", "Dig", "DoNothing") to step()
# and show the map with env.render()

env = gym.make('gym-forestfire-v0')
env.render()
print("Success!")