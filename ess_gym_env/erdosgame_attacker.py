"""The same environment, except the action space is for the attacker, not the defender."""
# THIS IS ENTIRELY UNTESTED.

import logging
import math
import gym
from .erdosgame import ErdosGameEnv
from gym import spaces
from gym.utils import seeding
import numpy as np
import random

logger = logging.getLogger(__name__)

# main game code


class Defender(object):
    """Shared interface for all defenders."""

    def __init__(self, K):
        """Add any other context the defender needs to choose."""
        self.K = K

    def pick_destroy(self, set1, set2):
        raise NotImplementedError('Must be implemented by subclass.')


class RandomDefender(Defender):
    """Pick set randomly."""

    def pick_destroy(self, set1, set2):
        # Maybe this should take a random seed?
        if np.random.random_sample() < 0.5:
            return 0
        else:
            return 1


class OptimalDefender(object):
    def pick_destroy(self, set1, set2):
        raise NotImplementedError('Make this later.')


class ErdosGameAttackerEnv(ErdosGameEnv):
    def __init__(self, *args, **kwargs):
        """Takes same function signature, plus a defender object."""
        super(ErdosGameAttackerEnv, self).__init__(*args, **kwargs)
        self.defender_object = kwargs.get('defender')
        if self.defender_object is None:
            # Default defender is random.
            self.defender_object = RandomDefender(self.K)

    def _step(self, action):
        """TODO."""
        self.update_game_state(action)
        return self.game_state.copy(), self.reward, \
            self.done, {"steps": self.steps, "visited": self.hash}

    def update_game_state(self, action):
        """
        Takes action, converts to sets, asks defender which to kill, sets class
        variables to propogate information to _step.
        """
        # record game state
        if self.hash != None:
            try:
                self.hash[self.game_state] += 1
            except KeyError:
                pass
        # record
        self.steps += 1
        set1, set2 = self.convert_action(action)
        defender_reply = self.defender_object.pick_destroy(set1, set2)
        if defender_reply == 0:
            destroy = set1
            push = set2
        else:
            destroy = set2
            push = set1

        self.game_state -= (destroy + push)
        push = push[:-1]
        push = np.insert(push, 0, 0)
        self.game_state += push
        assert(np.all(self.game_state >= 0)), print(
            "State negative!", self.game_state)
        if self.game_state[-1] > 0:
            self.attacker += 1
            # self.past_rewards.append(-1)
            self.reward = 1
            self.done = True

        if np.all(self.game_state == 0):
            self.defender += 1
            # self.past_rewards.append(1)
            self.reward = -1
            self.done = True

    def convert_action(self, action):
        """Decide how to convert action into final sets."""
        pass

    def _reset(self):
        """Do things need to change for adversarial case?"""
        self.game_state = self.observation_space.sample().astype("int")
        self.done = False
        self.reward = 0
        self.steps = 0
        # Avoid potential crazy bugs by copying state.
        return self.game_state.copy()
