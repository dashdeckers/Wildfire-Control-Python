# Wildfire-Control-Python
The custom OpenAI Gym environment simulates the spread of fire from a 2D, birds-eye-view perspective. 
To contain the fire, the agent (a bulldozer) should learn to dig a road around the fire, enclosing it completely. 
By doing so, we take away the fuel that the fire needs to spread further.\
Follow these instructions with python 3.6 in a virtual environment!

## Environment Usage:

### Install the gym environment:
`git clone https://github.com/dashdeckers/Wildfire-Control-Python`\
`cd Wildfire-Control-Python`\
`pip install -e gym-forestfire/.`

### Play the environment in human mode:
`make`\
`python`\
`import gym, gym_forestfire`\
`from misc import run_human`\
`sim = gym.make('gym-forestfire-v0')`\
`run_human(sim)`

## Reinforcement Learning (DQN) Usage:

### Install dependencies:
`pip install -r requirements.txt`

### Let the DQN learn and then let it play:
`python -i main.py`

Optionally pre-initialize memory buffer with generated memories (see code for details):\
`DQN.collect_memories(100)`

`DQN.learn(10000)`\
`DQN.play_optimal()`
