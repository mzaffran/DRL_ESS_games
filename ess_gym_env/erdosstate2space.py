import numpy as np

import gym

class ErdosState(gym.Space):
  """
  State type for Erdos game. Input looks like spaces.ErdosState(K, potential, weights, dist_probs, high_one_prob)
  K is of type int, giving the number of levels, potential the difficulty of the game
  and weights just to recycle computation, giving the number of weights for each level
  for use in sampling. Remainder are probabilities used for sampling
  """
  
  def __init__(self, K, potential, weights, unif_prob, geo_prob, diverse_prob,
         state_unif_prob, high_one_prob, geo_high, 
         unif_high, geo_ps, train_attacker, cattacker):
    
    assert K <= 40, "K is too large! Computing 2^-K may be very slow and inaccurate"
    
    self.K = K
    self.potential = potential
    self.weights = weights
    self.geo_prob = geo_prob
    self.unif_prob = unif_prob
    self.diverse_prob = diverse_prob
    self.state_unif_prob = state_unif_prob
    self.high_one_prob = high_one_prob
    self.geo_high = geo_high
    self.unif_high = unif_high
    self.geo_ps = geo_ps
    self.train_attacker = train_attacker
    self.cattacker = cattacker

    self.states_table = None
    self.all_states_table = None

  def potential_fn(self, state):
    return np.sum(state*self.weights)

  def sample(self):
    """
    Samples a random start state based on initialization configuration
    """
    
    # pick sample type according to probability
    samplers = ["unif", "geo", "diverse", "state_unif"]
    sample_idx = np.random.multinomial(1, [self.unif_prob, self.geo_prob, self.diverse_prob, self.state_unif_prob])
    idx = np.argmax(sample_idx)
    sampler = samplers[idx]
    
    if sampler == "unif":
      return self.unif_sampler()
    if sampler == "geo":
      return self.geo_sampler()
    if sampler == "diverse":
      return self.diverse_sampler()    
    if sampler == "unif_state":
      return self.state_unif_sampler()

  def get_high_one(self, state):
    """
    Takes in state and adds one piece at a high level
    """
    non_zero_idxs = [-2, -3, -4]
    idx_idxs = np.random.randint(low=0, high=3, size=10)
    for idx_idx in idx_idxs:
      non_zero_idx = non_zero_idxs[idx_idx]
      if self.potential_fn(state) + self.weights[non_zero_idx] <= self.potential:
        state[non_zero_idx] += 1
        break
    return state  
   
  def unif_sampler(self):
    """
    Samples pieces for states uniformly, for levels 0 to self.unif_high
    """
    state = np.zeros(self.K+1, dtype=int)
     
    # adds high one according to probability
    high_one = np.random.binomial(1, self.high_one_prob)
    if high_one:
      state = self.get_high_one(state)

    # checks potential of state, returning early if necessary
    if (self.potential -  self.potential_fn(state)) <= 0:
      return state
     
    # samples according to uniform probability
    pot_state = self.potential_fn(state)

    for i in range(max(10, int(1/(100000*self.weights[0])))):
      levels = np.random.randint(low=0, high=self.unif_high, size=int(np.min([100000, 1.0/self.weights[0]])))
      # adds on each level as the potential allows
      for l in levels:
        if pot_state + self.weights[l] <= self.potential:
          state[l] += 1
          pot_state += self.weights[l]
         
        # checks potential to break
        if pot_state >= self.potential - max(1e-8, self.weights[0]):
          break
      # checks potential to break
      if pot_state >= self.potential - max(1e-8, self.weights[0]):
        break
      
    return state
      
  def geo_sampler(self):
    """
    Samples pieces for states with geometric distributions, for levels 0 to self.geo_high
    and buckets them in from lowest level to highest level
    """
    state = np.zeros(self.K+1, dtype=int)
     
    # adds high one according to probability
    high_one = np.random.binomial(1, self.high_one_prob)
    if high_one:
      state = self.get_high_one(state)
    
    # pick the p in Geometric(p), where p is randomly chosen from predefined list of ps
    ps = self.geo_ps
    p_idx = np.random.randint(low=0, high=len(ps))
    p = ps[p_idx]
    for i in range(max(1000, int(1/(100000*self.weights[0])))):
      # get pieces at different levels, highest level = self.geo_high
      assert self.K+1 < 30, "K too high, cannot use geo sampler"
      levels = np.random.geometric(p, int(1.0/self.weights[0])) - 1
      idxs = np.where(levels < self.geo_high)
      levels = levels[idxs]
      
      # bin the levels into the same place which also sorts them from 0 to K
      # counts created separately to ensure correct shape
      tmp = np.bincount(levels)
      counts = np.zeros(self.K + 1)
      counts[:len(tmp)] = tmp
      
      # add levels to state with lowest levels going first
      for l in range(self.K + 1):
        max_pieces = (self.potential - self.potential_fn(state))/self.weights[l]
        max_pieces = int(np.min([counts[l], max_pieces]))
        state[l] += max_pieces
        
        # checks potential to break
        if self.potential_fn(state) >= self.potential - max(1e-8, self.weights[0]):
          break
      # checks potential to break
      if self.potential_fn(state) >= self.potential - max(1e-8, self.weights[0]):
        break
      
    return state
  
  def simplex_sampler(self, n):
    """ Samples n non-negative values between (0, 1) that sum to 1
    Returns in sorted order. """
    
    # edge case: n = 1
    if n == 1:
      return np.array([self.potential])

    values = [np.random.uniform() for i in range(n-1)]
    values.extend([0,1])
    values.sort()
    values_arr = np.array(values)
    
    xs = values_arr[1:] - values_arr[:-1]

    # return in decresing order of magnitude, to use for higher levels
    xs = self.potential*np.sort(xs)
    xs = xs[::-1]
    return xs    


  def diverse_sampler(self):
    """
    Tries to sample state to increase coverage in state space. Does this with three steps
    Step 1: Uniformly samples the number of non-zero idxs
    Step 2: Gets a set of idxs (between 0 to K-2) with size the number of nonzero idxs
        in Step 1
    Step 3: Divides up the potential available uniformly at random between the chosen idxs
    """
    
    # Sample number of nonzero idxs
    num_idxs = np.random.randint(low=1, high=self.K-1)

    # Sample actual idxs in state that are nonzero
    idxs = []
    all_states =[ i for i in  range(self.K - 1)] # can have nonzero terms up to state[K-2]
    for i in range(num_idxs):
      rand_id = np.random.randint(low=0, high=len(all_states))
      idxs.append(all_states.pop(rand_id))

    # sort idxs from largest to smallest to allocate
    # potential correctly
    idxs.sort()
    idxs.reverse()

    # allocate potential
    xs = self.simplex_sampler(num_idxs)

    # fill with appropriate number of pieces adding on any remaindr
    remainder = 0
    state = np.zeros(self.K+1, dtype=int)
    for i in range(num_idxs):
      idx = idxs[i]
      pot_idx = xs[i] + remainder
      num_pieces = int(pot_idx/self.weights[idx])
      state[idx] += num_pieces
      # update remainder
      remainder = pot_idx - num_pieces*self.weights[idx]

    return state

  def state_unif_sampler(self):
    """
  Sampler that draws a start state uniformly from self.states_table
  """
    assert self.states_table != None, "states_table attribute is not set, call enumerate_states_potential"

    high = len(self.states_table)
    idx = np.random.randint(low=0, high=high)
    state = np.array(self.states_table[idx]).astype("int")

    return state

  def enumerate_states_core(self, K, potential, N, weights):
    """
    This function takes in values for K, potential, N and weights
    and enumerates all states of that potential, returning them as
    a list.
    """ 

    # base case
    if K == 2:
      result = []
      max_N = np.floor(N*(weights[0]/weights[K-1])).astype("int") + 1
      
      for i in range(max_N):
        result.append([N - 2*i, i])

    # recursion
    else:
      result = []
      scaling = (weights[0]/weights[K-1])
      max_N = np.floor(N*scaling).astype("int") + 1
      
      for i in range(max_N):
        recursed_results = self.enumerate_states_core(K-1, potential-i*weights[K-1], int(N - i/scaling), weights[:-1])

        # edit recursed results and append
        for state in recursed_results:
          state.append(i)
        
        # add on to list of states
        result.extend(recursed_results)
    
    # NOTE: result contains list of states that are missing level K (which must always be 0)
    # this needs to be added on after getting the result
    return result
      

  def enumerate_states_potential(self):
    assert self.K <= 10, "K is too large for enumerating all states!"
    N = int(self.potential*np.power(2.0, self.K))
    raw_states = self.enumerate_states_core(self.K, self.potential, N, self.weights)
    
    # add 0 term corresponding to level K to raw states
    for state in raw_states:
      state.append(0)

    self.states_table = raw_states

  def enumerate_states_all(self, upperbound=1.0):
    """This function enumerates all states of all potentials
    from 0 to upper bound"""

    assert self.K < 10, "K is too large for enumerating all states"
    
    all_states = []
    for n in range(int(upperbound/self.weights[0])+1):
      N=n
      potential = float(n)*self.weights[0]
      raw_states = self.enumerate_states_core(self.K, potential, N, self.weights)

      # add on top 0 and append to results
      for state in raw_states:
        state.append(0)
      all_states.extend(raw_states)

    self.all_states_table = all_states      


  def contains(self, action):
    """
    TODO(maithra)
    """
    pass   
  
  
  @property
  def shape(self):
    if self.train_attacker:
      return (self.K+1,)
    else:
      return (2*self.K + 2,)
  def __repr__(self):
    return "ErdosGame" + str(self.shape)