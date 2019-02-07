#!/usr/bin/env python3

##
#
# A simple example of using our framework to
# evaluate performance on a simple reach-avoid problem
#
##

import numpy as np
from pySTL import STLFormula
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.optimize import minimize

# The scenario we're interested in involves moving past an obstacle
# and eventually reaching a target region. First, we'll plot these regions for visualization
fig, ax = plt.subplots(1)
ax.set_xlim((0,12))
ax.set_ylim((0,12))

obstacle = Rectangle((3,4),2,2,color='red',alpha=0.5)
target = Rectangle((7,8),1,1, color='green',alpha=0.5)

ax.add_patch(obstacle)
ax.add_patch(target)

###########################################################################
# System Definition                                                       #
###########################################################################

# To specify the system, we write this function which maps from an initial condition
# and a control sequence to an STL signal which we can check. The definition
# of this signal depends on the properties we're interested in. It might be the
# state:
#     s = [x_0, x_1, ... ,x_n]
# Or it might include the control as well:
#     s = [x_0, x_1, ... , x_n]
#         [u_0, u_1, ... , u_n]
# Or it might be another function of the state and control.

def signal(x_0,u):
    """
    Maps a control signal u to an STL signal we can check. Here we consider
    the signal to be the (x,y) position of a robot with double integrator dynamics. 

    Arguments:
        x_0 : a (n,1) numpy array representing the system's initial state
        u   : a (m,T) numpy array 
    """
    # System definition: x_{t+1} = A*x_t + B*u_t
    A = np.array([[1,1,0,0],[0,1,0,0],[0,0,1,1],[0,0,0,1]])
    B = np.array([[0,0],[1,0],[0,0],[0,1]])
    
    T = u.shape[1]      # number of timesteps

    # Pre-alocate the signal
    s = np.zeros((2,T)) 

    # Run the controls through the system and see what we get
    x = x_0
    for t in range(T):
        # extract the first and third elements of x
        s[:,t] = x[[0,2],:].flatten()   

        # Update the system state
        x = A@x + B@u[:,t][:,np.newaxis]   # ensure u is of shape (2,1) before applying

    return s


###########################################################################
# STL Specification                                                       #
###########################################################################

# Now we define the property we're interested in:
#   ALWAYS don't hit the obstacle and EVENTUALLY reach the goal in 20 steps.

# We'll assume the signal that our specification is over is a list of x,y coordinates
# at each timestep. We'll build up our specification from predicates, with the use of 
# this handy helper function:

def in_rectangle_formula(xmin,xmax,ymin,ymax):
    """
    Returns an STL Formula denoting that the signal is in
    the given rectangle.
    """
    # These are essentially predicates, since their robustnes function is
    # defined as (mu(s_t) - c) or (c - mu(s_t))
    above_xmin = STLFormula(lambda s, t : s[t,0] - xmin)
    below_xmax = STLFormula(lambda s, t : -s[t,0] + xmax)
    above_ymin = STLFormula(lambda s, t : s[t,1] - ymin)
    below_ymax = STLFormula(lambda s, t : -s[t,1] + ymax)

    # above xmin and below xmax ==> we're in the right x range
    in_x_range = above_xmin.conjunction(below_xmax)
    in_y_range = above_ymin.conjunction(below_ymax)

    # in the x range and in the y range ==> in the rectangle
    in_rectangle = in_x_range.conjunction(in_y_range)

    return in_rectangle

hit_obstacle = in_rectangle_formula(3,5,4,6)
at_goal = in_rectangle_formula(7,8,8,9)

obstacle_avoidance = hit_obstacle.negation().always(0,20)
reach_goal = at_goal.eventually(0,20)

full_specification = obstacle_avoidance.conjunction(reach_goal)

###########################################################################
# Optimization                                                            #
###########################################################################

# Here we'll define a cost function that ensures satisfaction of the
# specification, and optimize over it

def cost_function(u, m, x_0, specification, signal):
    """
    Defines a cost function over the control sequence u such that
    the optimal u maximizes the robustness degree of the specification.

    Arguments:
        u             : a (m*T,) numpy array representing a tape of control inputs
        m             : the dimension of the control input u
        x_0           : a (n, 1) numpy array representing the initial state
        specification : a STLFormula object representing the specification
        signal        : a function tha maps from u, x_0 to a signal used in the specification

    Returns:
        J : a scalar value indicating the degree of satisfaction of the specification.
            (negative ==> satisfied)
    """
    # Reshape the control input to (mxT). Vector input is required for some optimization
    # libraries. 
    T = int(len(u)/m)
    u = u.reshape((m,T))

    s = signal(x_0,u)
    J = - specification.robustness(s.T,0)

    return J

# Now we'll use scipy to optimize this cost function: any one of a variety of
# methods might be used


# Some parameters
x_0 = np.array([0,0,0,0])[:,np.newaxis]
m = 2
T = 21

# initial guess
u_0 = np.zeros((m,T)).flatten()  # initial guess

# Run the minimization!
res = minimize(cost_function, u_0, 
        args=(m, x_0, full_specification, signal),
        method='Nelder-Mead')

print(res)

u_star = res.x.reshape(m,T)

rho1 = cost_function(u_star.flatten(), 2, x_0, full_specification, signal)
print(rho1)

###########################################################################
# Evaluating Optimal Trajectory                                           #
###########################################################################

# Finally, we'll plot the optimal trajectory to see how we did

trajectory = signal(x_0, u_star)
ax.scatter(trajectory[0,:],trajectory[1,:])
#plt.legend()
plt.title("Specification: Always avoid red and eventually reach green")

plt.show()
