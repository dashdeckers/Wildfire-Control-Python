from gym.envs.registration import register

register(
    id='gym_forestfire_v0',
    entry_point='gym_forestfire.envs:ForestFire',
)