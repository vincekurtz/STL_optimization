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
from optimizers import fast_bayesian, differential_evolution_method

# The detailed implementation of this scenario is defined here:
from example_scenarios import EitherOr

# initialize the example with an initial state
x0 = np.asarray([0,0,0,0])[:,np.newaxis]
sys = EitherOr(x0,deterministic=False)

# Set up and solve an optimization problem over u
u_guess = np.zeros((2,sys.T+1)).flatten()   # initial guess

start_time = time.time()
res = minimize(sys.cost_function, 
        u_guess,
        method=differential_evolution_method)
end_time= time.time()

u_opt = res.x.reshape((2,sys.T+1))

# Evaluate the Results
print("")
print("Computation Time: %0.5fs" % (end_time-start_time))
print("")
print("Total Robustness Score: %0.5f" % sys.rho(u_opt))
print("")
print("Robustness Breakdown: ")
print("    Control            : %0.5f" % sys.rho(u_opt, spec=sys.control_bounded))
print("    Obstacle Avoidance : %0.5f" % sys.rho(u_opt, spec=sys.obstacle_avoidance))
print("    Goal Reaching      : %0.5f" % sys.rho(u_opt, spec=sys.reach_goal))
print("    Subgoal Reaching   : %0.5f" % sys.rho(u_opt, spec=sys.intermediate_target))
print("")

print(repr(u_opt))

#u_opt = np.array([[ 0.59056117,  0.48984312,  0.58504316, -0.54808972, -0.26220731,
#         0.29784566, -0.46526147, -0.14659776, -0.18540569, -0.01001986,
#        -0.18377343, -0.5706731 ,  0.59107893, -0.25968545,  0.45732217,
#         0.09419687,  0.15966823, -0.20205673,  0.33279684,  0.10059328,
#         0.13312721],
#       [ 0.5570162 ,  0.45483403, -0.29248885,  0.24451359, -0.3490124 ,
#         0.51309951,  0.5234845 ,  0.10977541, -0.4038066 , -0.1525798 ,
#        -0.36442248, -0.2314939 , -0.10792779,  0.28928851,  0.58695634,
#        -0.51566141, -0.29402082, -0.49531901, -0.08726829,  0.13232388,
#         0.45352642]])

start_time = time.time()
mean, var = sys.estimate_cost(u_opt,Nsim=100)
print(time.time()-start_time)

print(mean,var)

# Plot the results
fix, ax = plt.subplots(1)
plt.title("Reach Blue then Green, avoiding Red")
for trial in range(10):
    sys.plot_trajectory(u_opt,ax)
plt.show()
