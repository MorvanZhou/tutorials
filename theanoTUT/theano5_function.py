# View more python tutorials on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

# 5 - theano.function
"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
from __future__ import print_function
import numpy as np
import theano
import theano.tensor as T

# activation function example
x = T.dmatrix('x')
s = 1 / (1 + T.exp(-x))    # logistic or soft step
logistic = theano.function([x], s)
print(logistic([[0, 1],[-1, -2]]))

# multiply outputs for a function
a, b = T.dmatrices('a', 'b')
diff = a - b
abs_diff = abs(diff)
diff_squared = diff ** 2
f = theano.function([a, b], [diff, abs_diff, diff_squared])
print( f(np.ones((2, 2)), np.arange(4).reshape((2, 2))) )

# default value and name for a function
x, y, w = T.dscalars('x', 'y', 'w')
z = (x+y)*w
f = theano.function([x,
                     theano.In(y, value=1),
                     theano.In(w, value=2, name='weights')],
                   z)
print(f(23, 2, weights=4))