from DQN_DUEL import DQN_DUEL
from DQN_SARSA import DQN_SARSA

class DQN_BOTH(DQN_SARSA, DQN_DUEL):
    def __init__(self, sim, name="no_name", verbose=True):
    	# Only needs to call __init__ from SARSA because it has already
    	# inherited the changed functions from both and the rest is the same
        DQN_SARSA.__init__(self, sim, name, verbose)
