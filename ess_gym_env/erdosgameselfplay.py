from .erdosgame import ErdosGameEnv
import numpy as np


class SelfPlayErdosGameEnv(ErdosGameEnv):
    # Call this periodically to set the weights to the current model weights.
    # The optimal set proposal will then automatically do the right thing.
    def __init__(self, *args, **kwargs):
        super(SelfPlayErdosGameEnv, self).__init__(*args, **kwargs)
        self.true_weights = np.copy(self.weights)

    def set_weights(self, weights):
        print('Setting weights to', weights)
        self.weights = weights

    def reset_weights(self):
        print('Resetting weights')
        self.weights = np.copy(self.true_weights)
