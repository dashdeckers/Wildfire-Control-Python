# Wildfire-Control-Python
The custom OpenAI Gym environment simulates the spread of fire from a 2D, birds-eye-view perspective. 
To contain the fire, the agent (a bulldozer) should learn to dig a road around the fire, enclosing it completely. 
By doing so, we take away the fuel that the fire needs to spread further.

## Usage:

### Install the gym environment (via terminal):
`cd gym-forestfire/`\
`pip install -e .`

### Create the environment:
`import gym, gym_forestfire`\
`sim = gym.make('gym-forestfire-v0')`

### Pass an action to the agent: 
`sim.step(action)`

### Valid actions are:
("N", "S", "E", "W", "D", ""), for:\
Go up, down, right and left, dig the current tile, do nothing.

### Show the environment:
`sim.render()`
