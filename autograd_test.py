#!/usr/bin/env python3

import time
import numpy as onp                     # original numpy
import jax.numpy as np                  # wrapped numpy
from jax import jacfwd, jacrev, grad    # automatic differentiation

# The detailed implementation of this scenario is defined here:
from example_scenarios import ReachAvoid

# initialize the example with an initial state
x0 = np.asarray([0,0])[:,np.newaxis]
example = ReachAvoid(x0,T=20)

# Set up and solve an optimization problem over u
u_guess = np.zeros((2,21)).flatten()   # initial guess
#u_guess = np.random.rand(u_guess.shape[0])

# Compute gradient of cost function
cost_function_grad = jacfwd(example.cost_function)

# Evaluate cost function at initial guess
st = time.time()
print(example.cost_function(u_guess))
print(time.time() - st)

# Evaluate gradient at initial guess
st = time.time()
print(cost_function_grad(u_guess))
print(time.time() - st)

st = time.time()
print(cost_function_grad(onp.random.rand(u_guess.shape[0])))
print(time.time() - st)

