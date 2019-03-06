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

# The detailed implementation of this scenario is defined here:
from example_scenarios import EitherOr

# A variety of optimization routines are defined in optimizers.py
from optimizers import cross_entropy, differential_evolution_method, basinhopping_method, gp_bayesian, fast_bayesian

def add_to_comparison(method, name, ax, options=None):
    """
    Evaluate the given optimization method on the EitherOr scenario,
    and plot the results. 

    Arguments:
        method  : a string or function given as the method argument to scipy.minimize
        name    : a string describing what this optimization method is
        ax      : the matplotlib axis to use
        options : a dictionary passed to the options field on scipy.minimize
    """
    # initialize the example with an initial state
    x0 = np.asarray([0.0,0,0,0])[:,np.newaxis]
    sys = EitherOr(x0,T=75)

    print("###########################################")
    print(name)
    print("###########################################")

    # Set up and solve the optimization problem
    u_guess = np.zeros((2,sys.T+1)).flatten()   # initial guess

    start_time = time.time()
    res = minimize(sys.cost_function, u_guess,
            method=method,
            options=options)
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

    # label with computation time and robustness degree
    label_text = "%s : %0.1fs, $\\rho$ = %0.2f" % (name, (end_time-start_time), sys.rho(u_opt))

    # Plot the results
    sys.plot_trajectory(u_opt, ax, label=label_text)

# Set up the plot
fig, ax = plt.subplots(1)

# Try out the different optimization methods

#add_to_comparison("nelder-mead", "Nelder-Mead", ax, options={'disp':True,'adaptive':False,'maxiter':20000})
#plt.legend()   # show plot in real time
#plt.pause(0.05) 

#add_to_comparison("nelder-mead", "Adaptive Nelder-Mead", ax, options={'disp':True,'adaptive':True,'maxiter':20000})
#plt.legend()
#plt.pause(0.05)

#add_to_comparison("bfgs", "BFGS (gradient-based)", ax, options={'disp':True,'maxiter':20000})
#plt.legend()
#plt.pause(0.05)

#add_to_comparison(basinhopping_method, "Basinhopping (sampling-based)", ax)
#plt.legend()
#plt.pause(0.05)

add_to_comparison(gp_bayesian, "Bayesian Optimization (GP)", ax)
#plt.legend()
#plt.pause(0.05)

#add_to_comparison(fast_bayesian, "DE + BO", ax)

#add_to_comparison(cross_entropy, "Cross-entropy Optimization", ax, options={'disp':True,'niter':200})
#plt.legend()
#plt.pause(0.05)

#add_to_comparison(differential_evolution_method, "Differential Evolution", ax)
#plt.legend()
#plt.pause(0.05)

#add_to_comparison("powell", "Powell", ax, options={'disp':True})
#plt.legend()
#plt.pause(0.05)


# Display the plot
ax.set_title("Visit Blue and Green, avoiding Red")
plt.show()
