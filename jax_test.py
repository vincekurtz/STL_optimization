#!/usr/bin/env python3

import jax.numpy as np  # thinly-wrapped numpy
from jax import grad, jacfwd, jacrev    # automatic differentiation
from jax.ops import index_update, index_add, index

def f(x):
    x = index_add(x, index[1,:], np.array([5.0]))
    return x.T@x

grad_f = jacrev(f)

x = np.array([[1.0],[2.0],[3.0],[4.0]])
print(f(x))
print(grad_f(x))

