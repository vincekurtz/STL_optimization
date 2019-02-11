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
from scipy.optimize import minimize, basinhopping, OptimizeResult
from skopt import gp_minimize

# The detailed implementation of this scenario is defined here:
from example_scenarios import EitherOr

# initialize the example with an initial state
x0 = np.asarray([0,0,0,0])[:,np.newaxis]
sys = EitherOr(x0)

# Defining custom optimization functions
def basinhopping_custom(fun, x0, args=(), **options):
    """
    Basinhopping, a simulated annealing-like approach
    """
    res = basinhopping(fun, x0, disp=True)
    return res

def gp_bayesian(fun, x0, args=(), **options):
    """
    Baysian optimization with Gaussian Processes
    """
    dimensions = [(-0.9,0.9) for i in range(len(x0))]  # set limits of search space
    res = gp_minimize(fun, dimensions, 
                                verbose=True, 
                                acq_func="EI",
                                noise=1e-9)
    res.x = np.asarray(res.x)  # ensures output is a numpy array
    return res

# Set up and solve an optimization problem over u
u_guess = np.zeros((2,21)).flatten()   # initial guess

start_time = time.time()
res = minimize(sys.cost_function, u_guess,
        method=gp_bayesian,
        options={
                    'disp':True,
                    'adaptive':True,
                    'maxiter':20000,
                    'ftol':1e-6,
                    'xtol':1e-6
                }
        )
end_time= time.time()

u_opt = res.x.reshape((2,21))

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


# Plot the results
fix, ax = plt.subplots(1)
plt.title("Reach Blue then Green, avoiding Red")
sys.plot_trajectory(u_opt,ax)
plt.show()
