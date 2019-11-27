#!/usr/bin/env python3

import time
import autograd.numpy as np  # wrapped numpy
from autograd import grad    # symbolic differentiation

# The detailed implementation of this scenario is defined here:
from example_scenarios import ReachAvoid

# initialize the example with an initial state
x0 = np.asarray([0,0])[:,np.newaxis]
example = ReachAvoid(x0,T=20)

# Set up and solve an optimization problem over u
u_guess = np.zeros((2,21)).flatten()   # initial guess
#u_guess = np.random.rand(u_guess.shape[0])

# Compute gradient of cost function
cost_function_grad = grad(example.cost_function)

# Evaluate cost function at initial guess
st = time.time()
J = example.cost_function(u_guess)
print("cost: %s" % J)
print("compute time: %s" % (time.time()-st))

# Evaluate gradient at initial guess
st = time.time()
dJdu = cost_function_grad(u_guess)
print("gradient: %s" % dJdu)
print("compute time: %s" % (time.time()-st))
#print(cost_function_grad(onp.random.rand(u_guess.shape[0])))

