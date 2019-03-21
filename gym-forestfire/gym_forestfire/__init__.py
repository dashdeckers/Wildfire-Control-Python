from gym.envs.registration import register

register(
    id='gym-forestfire-v0',
    entry_point='gym_forestfire.envs:ForestFire',
)