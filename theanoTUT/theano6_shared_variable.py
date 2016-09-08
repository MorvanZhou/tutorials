# View more python tutorials on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

# 6 - shared variables
"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
from __future__ import print_function
import numpy as np
import theano
import theano.tensor as T

state = theano.shared(np.array(0,dtype=np.float64), 'state') # inital state = 0
inc = T.scalar('inc', dtype=state.dtype)
accumulator = theano.function([inc], state, updates=[(state, state+inc)])

# to get variable value
print(state.get_value())
accumulator(1)   # return previous value, 0 in here
print(state.get_value())
accumulator(10)  # return previous value, 1 in here
print(state.get_value())

# to set variable value
state.set_value(-1)
accumulator(3)
print(state.get_value())

# temporarily replace shared variable with another value in another function
tmp_func = state * 2 + inc
a = T.scalar(dtype=state.dtype)
skip_shared = theano.function([inc, a], tmp_func, givens=[(state, a)]) # temporarily use a's value for the state
print(skip_shared(2, 3))
print(state.get_value()) # old state value
