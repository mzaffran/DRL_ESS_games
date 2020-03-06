# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %% [markdown]
# ## This notebook trains a defender agent with PPO
# 
# This notebook offers example code on how to train a defender agent on the ESS environment with PPO. Note that for the code to work correctly, you'll need the modified versions of gym and OpenAI baselines installed (we recommend on a virtual environment). 
# 
# Links to modified gym/baselines:
# 
# 
# https://github.com/rubai5/baselines
# 
# 
# https://github.com/rubai5/gym
# 
# %% [markdown]
# ### Imports

# %%
import os, sys
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from datetime import datetime
import pickle

from mpi4py import MPI
import os.path as osp
import gym, logging
import baselines
from baselines import logger
from baselines.ppo1 import pposgd_simple_generalization, mlp_policy
import baselines.common.tf_util as U
from copy import deepcopy

# %% [markdown]
# ### Game Paramters
# The ESS game has a huge number of possible states. The gym environment has some ways of sampling from these states, and here, we set the parameters to mix the distributions as desired

# %%
# parameters
name = "ErdosGame-v0"
seed = 101

# game specific parameters
K = 15
potential = 0.9

# sampling probabilities, must sum to 1
unif_prob = 0.0
geo_prob = 1.0
diverse_prob = 0.0
state_unif_prob = 0.0 # can only use if K is small < 10 -- try to use previous methods instead

assert (unif_prob + geo_prob + diverse_prob + state_unif_prob == 1), "probabilites don't sum to 1"

# attacker plays adversarially?
adverse_set_prob = 0.0
disj_supp_prob = 0.5

# high one
high_one_prob = 0.0

# upper limits for start state sampling
geo_high = K - 2
unif_high = max(3, K-3)

# putting into names_and_args argument
names_and_args = {"K" : K, 
                  "potential" : potential, 
                  "unif_prob" : unif_prob, 
                  "geo_prob" : geo_prob,
                  "diverse_prob" : diverse_prob, 
                  "state_unif_prob" : state_unif_prob, 
                  "high_one_prob" : high_one_prob, 
                  "adverse_set_prob" :adverse_set_prob, 
                  "disj_supp_prob" : disj_supp_prob, 
                  "geo_high" : geo_high, 
                  "unif_high" :unif_high }

# %% [markdown]
# ### Model Paramters

# %%
HID_SIZE=300
NUM_HID_LAYERS=2

# %% [markdown]
# ### Policy Net, Train and Test function

# %%
# functions to initialize environment and train model

def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space, 
                                    hid_size=HID_SIZE, num_hid_layers=NUM_HID_LAYERS)
    
def make_policies(ob_space, ac_space, policy_func):
    pi = policy_func("pi", ob_space, ac_space)
    oldpi = policy_func("old_pi", ob_space, ac_space)
    return pi, oldpi

def train(env_train, pi, oldpi, names_and_args, num_timesteps, test_envs):
    #workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    #set_global_seeds(workerseed)
    
    env_train.reset()
    if test_envs:
        for test_env in test_envs:
            test_env.reset()
    
    #env.seed(workerseed)
    gym.logger.setLevel(logging.WARN)
    # create file handler which logs even debug messages
    fh = logging.FileHandler('spam.log')
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    gym.logger.addHandler(fh)
    gym.logger.addHandler(ch)
        

    policy_net, info = pposgd_simple_generalization.learn(env_train, pi, oldpi,
        max_timesteps=num_timesteps,
        timesteps_per_batch=100,
        clip_param=0.2, entcoeff=0.01,
        optim_epochs=4, optim_stepsize=1e-3, optim_batchsize=50,
        gamma=0.99, lam=0.95,
        schedule='linear',
        test_envs=test_envs
    )

    return policy_net, info


def test_policy(num_rounds, policy_net, test_env):
    total_reward = 0.0
    horizon = test_env.observation_space.K*num_rounds # generate around num_rounds draws
    seg_gen = pposgd_simple_generalization.traj_segment_generator(policy_net, test_env, horizon, stochastic=True)
    
    # call generator
    results = seg_gen.__next__()
    mean_reward = np.mean(results["ep_rets"])
    actions = results["ac"]
    labels = results["label"]
    mean_correct_actions = compute_correct_actions(labels, actions)
    return mean_reward, mean_correct_actions

def compute_correct_actions(label, ac):
    count = 0
    idxs = np.all((label == [1,1]), axis=1)
    count += np.sum(idxs)
    new_label = label[np.invert(idxs)]
    new_ac = ac[np.invert(idxs)]
    count += np.sum((new_ac == np.argmax(new_label, axis=1)))
    avg = count/len(label)
    return avg

# %% [markdown]
# ### A note on sessions
# To run most of the baselines code, we need to explicitly state that the session is the default one, i.e. start with
#     <code here>
#     with sess.as_default():
#     </code here>
# The code is currently set up for initializing sess = U.single_threaded_session() as a global variable and closing/reseting the graph explicitly to enable restarts, etc. Note that U.reset() must be used along with tf.reset_default_graph()
# %% [markdown]
# ### Functions to load graphs and sessions

# %%
# utilities
def reset_session_and_graph():
    try:
        sess.close()
    except:
        pass
    tf.reset_default_graph()
    U.reset()
    
def save_session(fp):
    # saves session
    assert fp[-5:] == ".ckpt", "checkpoint name must end with .ckpt"
    saver = tf.train.Saver()
    saver.save(sess, fp)
    
def load_session_and_graph(fp_meta, fp_ckpt):
    saver = tf.train.import_meta_graph(fp_meta)
    saver.restore(sess, fp_ckpt)
    U.load_state(fp_ckpt)


# %%
# Train network over a number of repeats
import json
import os
from copy import deepcopy
repeats = 3
SAVE_FP = "./tmp/models/"
os.makedirs(SAVE_FP, exist_ok=True)

from collections import defaultdict

results = defaultdict(lambda: defaultdict(lambda: dict()))

from tqdm import tqdm
test_num_rounds = 100
for K in [10]:  # Fixed to 10
    for potential in [.99]:
        for dis_adv_prob in tqdm(np.linspace(0, 1 , 20+1)):
            names_and_args["K"] = K
            names_and_args["geo_high"] = K-2
            names_and_args["unif_high"] = max(3, K-3)
            names_and_args["potential"] = potential
            names_and_args["disj_supp_prob"] = dis_adv_prob
            rewards = list()
            test_rewards = list()
            for rep in range(repeats):
                reset_session_and_graph()
                sess = U.single_threaded_session()
                with sess.as_default():
                    erdos_env = gym.make(name, **names_and_args)
                    pi, oldpi = make_policies(erdos_env.observation_space, 
                                              erdos_env.action_space, 
                                              policy_fn)                
                    pi, info = train(erdos_env, pi, oldpi, 
                                     names_and_args, 
                                     num_timesteps=50000, 
                                     test_envs=list())  # Add test environnements maybe?
                    rewards.append(info["rewards"])

                    # Test policy
                    for test_env_adv_prob in tqdm(np.linspace(0, 1 , 20+1)):
                        args = deepcopy(names_and_args)
                        args["disj_supp_prob"] = test_env_adv_prob
                        test_env = gym.make(name, **args)
                        results[dis_adv_prob][test_env_adv_prob][rep] = \
                            test_policy(test_num_rounds, pi, test_env)
                    with open("results.json", "w") as file:
                        file.write(json.dumps(results))
                    
                    # Save model
                    model_fp = "{}model_K_{}_potential_{}_rep_{}_disj_{}.ckpt".format(
                        SAVE_FP, K, potential, rep, dis_adv_prob)
                    save_session(model_fp)

            # Save results
            with open(SAVE_FP+"rewards_K%02d_potential%f.p"%(K, potential), "wb") as f:
                pickle.dump(rewards, f)


# %%


