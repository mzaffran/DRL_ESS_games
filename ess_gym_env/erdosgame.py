# gym like version

# imports for creating game
import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import random

logger = logging.getLogger(__name__)

# main game code


class ErdosGameEnv(gym.Env):

    def __init__(self, K, potential, unif_prob,
                 geo_prob, diverse_prob, state_unif_prob, high_one_prob,
                 adverse_set_prob, disj_supp_prob, geo_high, unif_high,
                 train_attacker=False, cattacker=False,
                 geo_ps=[0.45, 0.5, 0.6, 0.7, 0.8], hash_states=None):

        self.K = K
        self.unif_prob = unif_prob
        self.geo_prob = geo_prob
        self.diverse_prob = diverse_prob
        self.state_unif_prob = state_unif_prob
        assert(self.state_unif_prob + self.diverse_prob +
               self.geo_prob + self.unif_prob == 1), \
            "State generation probabilities do not add to 1"

        self.high_one_prob = high_one_prob
        self.train_attacker = train_attacker
        self.cattacker = cattacker

        self.adverse_set_prob = adverse_set_prob
        self.disj_supp_prob = disj_supp_prob
        assert(self.adverse_set_prob + self.disj_supp_prob <= 1), \
            "Attacker strategy probabilities do not add to 1"

        self.geo_high = geo_high
        self.unif_high = unif_high
        self.geo_ps = geo_ps
        self.hash = hash_states

        # adding an attribute called model_state to pass in values to
        self.model_state = []

        # build up idxs to deal with large powers of 2
        idxs = range(self.K + 1)

        self.random = random
        self.weights = np.power(2.0, [-(self.K - i) for i in idxs])
        self.attacker = 0
        self.defender = 0
        self.steps = 0
        self.potential = potential

        # gym specific stuff
        sample_dict = {"K": K, "potential": potential, "weights": self.weights,
                       "unif_prob": unif_prob, "geo_prob": geo_prob, "diverse_prob": diverse_prob,
                       "state_unif_prob": state_unif_prob, "high_one_prob": high_one_prob,
                       "geo_high": geo_high, "unif_high": unif_high, "geo_ps": geo_ps,
                       "train_attacker": train_attacker, "cattacker": cattacker}

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.ErdosState(**sample_dict)
        self.viewer = None

        # game_state is the state of the game, state is [A, B]
        self.game_state = None
        self.state = None

        # Maithra: commenting out seed for now
        # self._seed()

    def potential_fn(self, state):
        return np.sum(state*self.weights)

    def set_model_state(self, model_state):
        self.model_state = model_state

    def _step(self, action):
        """
        Updates state and returns new state, reward, done, and any info (as a dict)
        """
        # update state and determine if game ends
        # return empty log for now
        self.update_game_state(action)
        if self.done:
            # return state as is without picking new sets as this state will be discarded
            return self.state, self.reward, self.done, {"steps": self.steps, "visited": self.hash}

        # otherwise pick next sets to get new updated state and game state
        else:
            A, B = self.propose_sets()
            self.state = np.concatenate([A, B])
            return self.state, self.reward, self.done, {"steps": self.steps, "visited": self.hash}

    def update_game_state(self, action):
        """
        Function updates game_state, using state  (which is concat of [A, B])
        """
        # Get sets from game state
        A = self.state[: self.K + 1]
        B = self.state[self.K + 1:]

        # assert that sets have been correctly chosen
        assert(np.all(self.game_state - A - B >= 0)), print("Constraint broken, difference, state, A, B",
                                                            self.game_state - A - B, self.game_state, A, B)

        # record game state
        if self.hash != None:
            try:
                self.hash[self.game_state] += 1
            except KeyError:
                pass

        # record
        self.steps += 1

        if action == 0:
            destroy = A
            push = B
        else:
            destroy = B
            push = A

        self.game_state -= (destroy + push)
        push = push[:-1]
        push = np.insert(push, 0, 0)
        self.game_state += push

        assert(np.all(self.game_state >= 0)), print(
            "State negative!", self.game_state)

        if self.game_state[-1] > 0:
            self.attacker += 1
            # self.past_rewards.append(-1)
            self.reward = -1
            self.done = True

        if np.all(self.game_state == 0):
            self.defender += 1
            # self.past_rewards.append(1)
            self.reward = 1
            self.done = True

    def propose_sets(self):
        # picks optimal method of proposing sets
        # with probability self.difficulty, otherwise
        # picks sets at random

        # if only few pieces left, play optimally
        num_idxs = np.sum(self.game_state > 0)
        if num_idxs <= 3:
            #logging.info("proposing with opt")
            A, B = self.propose_sets_opt()

        else:
            propose_types = ["adversarial", "disj_support", "opt"]
            idx_arr = np.random.multinomial(1,
                                            [self.adverse_set_prob, self.disj_supp_prob,
                                             1 - self.adverse_set_prob - self.disj_supp_prob])
            idx = np.argmax(idx_arr)
            propose_type = propose_types[idx]

            if propose_type == "adversarial":
                #logging.info("proposing with adversarial")
                assert self.model_state != [
                ], "cannot propose adversarial sets with empty model state - call set_model_state() first"
                A, B = self.propose_sets_adversarial()
            elif propose_type == "disj_support":
                #logging.info('proposing with disj support')
                A, B = self.propose_sets_disj_support()
            elif propose_type == "opt":
                A, B = self.propose_sets_opt()
            else:
                raise ValueError("unsupported set propose_type")

        return A, B

    def equal_divide(self, A, B, potA, potB, l, l_weight, weight, l_pieces):
        # divides up pieces when potA, potB are equal except off by 1

        if l_pieces == 0:
            return A, B

        if l_pieces % 2 == 0:
            A[l] += l_pieces/2
            B[l] += l_pieces/2

        else:
            larger = np.ceil(l_pieces/2)
            smaller = np.floor(l_pieces/2)
            assert larger + smaller == l_pieces, print("division incorrect",
                                                       larger, smaller, l_pieces)

            if potA < potB:
                A[l] += larger
                B[l] += smaller

            elif potB < potA:
                A[l] += smaller
                B[l] += larger

            else:
                prob_A = np.random.binomial(1, 0.5)
                if prob_A:
                    A[l] += larger
                    B[l] += smaller
                else:
                    A[l] += smaller
                    B[l] += larger

        return A, B

    def propose_sets_disj_support(self):
        # proposes sets of disjoint support
        # with varying potential split

        A = np.zeros(self.K+1, dtype=int)
        B = np.zeros(self.K+1, dtype=int)

        nonzeros = np.where(self.game_state > 0)[0]
        thresholds = [1./3, 5./16, 14./32]
        _ = np.random.multinomial(3, [0.8, 0.1, 0.1])
        _ = np.argmax(_)
        threshold = thresholds[_]
        idxs = nonzeros[np.random.permutation(len(nonzeros))]

        potA = self.potential_fn(A)
        potB = self.potential_fn(B)
        for idx in idxs:
            l_pieces = self.game_state[idx]
            # check to see what potential of pieces is
            # if potential very large, fraction, equally divide
            if l_pieces*self.weights[idx] >= self.potential/2.:
                # try to equally divide
                if l_pieces % 2 == 0:
                    pieces = int(l_pieces/2)
                    A[idx] += pieces
                    B[idx] += pieces
                    potA += (l_pieces*self.weights[idx])/2.
                    potB += (l_pieces*self.weights[idx])/2.
                else:
                    A[idx] += int(l_pieces/2)
                    B[idx] += (int(l_pieces/2) + 1)
                    potA += int(l_pieces/2)*self.weights[idx]
                    potB += (int(l_pieces/2) + 1)*self.weights[idx]

            else:
                if potA >= threshold*self.potential:
                    B[idx] += l_pieces
                    potB += l_pieces*self.weights[idx]
                else:
                    A[idx] += l_pieces
                    potA += l_pieces*self.weights[idx]
        # vary which of A or B is underweighted set
        p = np.random.uniform(low=0, high=1)
        if p >= 0.5:
            return B, A
        else:
            return A, B

    def propose_sets_adversarial(self):
        # proposes adversarial sets for current
        # weight of model

        A = np.zeros(self.K+1, dtype=int)
        B = np.zeros(self.K+1, dtype=int)

        diff = (self.game_state*(self.weights -
                                 (self.model_state/np.abs(self.model_state[-2]))))[:-1]

        # want to fill up set with most underweighted terms first
        idxs = np.argsort(diff)[::-1]
        idxs = np.append(idxs, self.K)

        # fill up set to get around half of existing potential
        threshold = self.potential_fn(self.game_state)/2

        for i in idxs:

            # check potential to break
            potA = self.potential_fn(A)
            if potA > threshold + max(self.weights[0], 1e-8):
                break

            # get the number of pieces
            l_pieces = self.game_state[0, i]
            if l_pieces == 0:
                continue
            # add to A
            num_pieces = np.ceil(
                (threshold + max(self.weights[0], 1e-8) - potA)/self.weights[i])
            A[i] += np.min([l_pieces, num_pieces])

        # B is the complement of A
        B = self.game_state - A
        assert (np.all(B >= 0)), print("state, A and B", self.game_state, A, B)

        # vary which of A or B is underweighted set
        p = np.random.uniform(low=0, high=1)
        if p >= 0.5:
            return B, A
        else:
            return A, B

    def propose_sets_opt(self):
        # proposes optimial choices of sets
        # by givng two sets with potential both
        # >= 1/2 (can do by splitting lemma)

        A = np.zeros(self.K+1, dtype=int)
        B = np.zeros(self.K+1, dtype=int)

        levels = [i for i in range(self.game_state.shape[0])]
        levels.reverse()

        for l in levels:
            l_pieces = self.game_state[l]
            if l_pieces == 0:
                #logging.info("skipping level %d"%l)
                continue

            weight = self.weights[l]
            l_weight = l_pieces*weight
            potA = self.potential_fn(A)
            potB = self.potential_fn(B)

            # divide equally at that level if potentials are equal
            if potA == potB:
                A, B = self.equal_divide(
                    A, B, potA, potB, l, l_weight, weight, l_pieces)

            # if potentials are not equal
            else:
                diff = np.abs(potA - potB)
                num_pieces = np.ceil(diff/weight).astype("int")

                # if the number of pieces which are the difference is less than l_pieces
                if num_pieces <= l_pieces:
                    diff_pieces = num_pieces

                    if potA < potB:
                        A[l] += diff_pieces
                    else:
                        B[l] += diff_pieces

                    l_pieces -= diff_pieces
                    A, B = self.equal_divide(
                        A, B, potA, potB, l, l_weight, weight, l_pieces)

                else:
                    if potA < potB:
                        A[l] += l_pieces
                    else:
                        B[l] += l_pieces

            #potA = self.potential_fn(A)
            #potB = self.potential_fn(B)

        return A, B

    def _reset(self):
        """
        Note: when playing adversarially, call set_model_state
        before reset
        """
        self.game_state = self.observation_space.sample().astype("int")
        self.done = False
        self.reward = 0
        self.steps = 0

        # get first observation here!!!
        A, B = self.propose_sets()
        self.state = np.concatenate([A, B])

        return self.state

    def ground_truth(self):
        potentials = []
        potentials.append(self.potential_fn(self.state[:self.K+1]))
        potentials.append(self.potential_fn(self.state[self.K+1:]))
        return np.array(potentials)

    def labels(self):
        potentials = self.ground_truth()
        set_labels = (potentials == np.max(potentials)).astype("int")
        return set_labels
