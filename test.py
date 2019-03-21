import gym
import gym_forestfire

# Usage: pass an action ("N", "S", "E", "W", "Dig") to step()
# and show the map with env.render()

env = gym.make('gym-forestfire-v0')
env.render()
print("Success!")
