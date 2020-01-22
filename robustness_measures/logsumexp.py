##
#
# This script defines a numerically stable, differentiable log-sum-exponential,
# which can be used to approximate max. 
#
##

import autograd.numpy as np
from autograd.extend import primitive, defvjp

@primitive
def logsumexp(x):
    max_x = np.max(x)
    y = np.zeros(2)
    y[1] = 2
    return max_x + np.log(np.sum(np.exp(x - max_x)))

def logsumexp_vjp(ans,x):
    x_shape = x.shape
    return lambda g: np.full(x_shape,g)*np.exp(x-np.full(x_shape,ans))

defvjp(logsumexp, logsumexp_vjp)
