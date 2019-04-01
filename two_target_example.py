#!/usr/bin/env python3

##
#
# This file contains a simple example of using STL robustness optimization
# to achive a specification that involves moving to one of two regions before
# eventually reaching a target. 
#
##

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from optimizers import fast_bayesian

# The detailed implementation of this scenario is defined here:
from example_scenarios import EitherOr

# initialize the example with an initial state
x0 = np.asarray([0,0,0,0])[:,np.newaxis]
example = EitherOr(x0)

# Set up and solve an optimization problem over u
u_guess = np.zeros((2,example.T+1)).flatten()   # initial guess
u_opt = u_guess.reshape((2,example.T+1))

#start_time = time.time()
#res = minimize(example.cost_function, 
#        u_guess,
#        method=fast_bayesian)
#end_time= time.time()
#
#u_opt = res.x.reshape((2,example.T+1))
#
## Evaluate the Results
#print("")
#print("Computation Time: %0.5fs" % (end_time-start_time))
#print("")
#print("Total Robustness Score: %0.5f" % example.rho(u_opt))
#print("")
#print("Robustness Breakdown: ")
#print("    Control            : %0.5f" % example.rho(u_opt, spec=example.control_bounded))
#print("    Obstacle Avoidance : %0.5f" % example.rho(u_opt, spec=example.obstacle_avoidance))
#print("    Goal Reaching      : %0.5f" % example.rho(u_opt, spec=example.reach_goal))
#print("    Subgoal Reaching   : %0.5f" % example.rho(u_opt, spec=example.intermediate_target))
#print("")

mu, var = example.estimate_cost(u_opt)

print(mu)
print(var)

# Plot the results
fix, ax = plt.subplots(1)
plt.title("Reach Blue then Green, avoiding Red")
for trial in range(10):
    u_opt = np.asarray([[0.05 for i in range(example.T+1)] for j in range(2)])
    example.plot_trajectory(u_opt,ax)

plt.show()
