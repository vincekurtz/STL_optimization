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
from scipy.optimize import minimize, differential_evolution
from skopt import gp_minimize
from optimizers import fast_bayesian, differential_evolution_method, gp_bayesian
from multiprocessing import Process, Manager

# The detailed implementation of this scenario is defined here:
from example_scenarios import EitherOr

# initialize the example with an initial state
x0 = np.asarray([0,0,0,0])[:,np.newaxis]
sys = EitherOr(x0)

########################################################################
# Find a few initial guesses with DE over the nominal system
########################################################################

de_restarts = 8
de_iter = 5

u_guess = []
bounds = [(-0.9,0.9) for i in range(2*(sys.T+1))]   # bounds on control input

print("\n===> Finding initial point with Differential Evolution\n")

# We'll use multithreading!
# calculate an initial guess and add it to a list
def target_func(lst, i):
    np.random.seed()   # need to ensure each thread gets a different random start
    res = differential_evolution(sys.cost_function, bounds, disp=True, maxiter=de_iter)
    lst.append(list(res.x))

with Manager() as manager:
    L = manager.list()   # this can be shared between processes
    processes = []
    for i in range(de_restarts):
        p = Process(target=target_func, args=(L, i))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    for i in range(de_restarts):  # put in nice list format
        u_guess.append(L[i])


#########################################################################
# Use Bayesian Opt to maximize P(rho>0) ==> minimize P(J>0)
#########################################################################

print("\n===> Improving the estimate with Bayesian Optimization\n")

res = gp_minimize(sys.probabilistic_cost_function,
                  bounds,
                  verbose=True,
                  n_calls=50,
                  acq_func="LCB",
                  kappa=0.01,
                  n_jobs=-1,
                  noise=1.0,
                  x0=u_guess)

res.x = np.asarray(res.x)  # make sure this is a np array
u_opt = res.x


guess_sat_prob = 1-min( [sys.probabilistic_cost_function(np.asarray(u),Nsim=100) for u in u_guess])
sat_prob = 1-sys.probabilistic_cost_function(u_opt,Nsim=100)

print("")
print("Satisfaction Probability of the Best Initial Guess: %s" % guess_sat_prob)
print("Final Satisfaction Probability: %s" % sat_prob)


# Plot the results
fix, ax = plt.subplots(1)
plt.title("Reach Blue then Green, avoiding Red")
for trial in range(50):
    sys.plot_trajectory(u_opt.reshape((2,sys.T+1)),ax)
plt.show()
