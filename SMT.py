#!/usr/bin/env python3

##
#
# An example of using Satisfiability Modulo Theories
# to find a satisfying trajectory.
#
##

import time
import numpy as np
import matplotlib.pyplot as plt
import z3

from example_scenarios import EitherOr

def mymax(a,b):
    """
    Special function to encode a max constraint in z3
    """
    return z3.If(a > b, a, b)

def real_to_float(z3_real):
    """
    Helper function to convert a real valued z3 variable to a python float. 
    """
    if z3.is_algebraic_value(z3_real):
        z3_real = z3_real.approx(20)  # approximate with precesion 1/10^20

    assert z3.is_rational_value(z3_real), "Z3 variable is not rational!"

    num = z3_real.numerator_as_long()
    den = z3_real.denominator_as_long()

    return float(num/den)

# Define system parameters
N = 10  # number of steps
x0 = np.asarray([0,0,0,0])[:,np.newaxis]
sys = EitherOr(x0,T=N-1)

# Define SMT real-valued variables for each control input
u1 = [z3.Real('u1_%s'%i) for i in range(N)]
u2 = [z3.Real('u2_%s'%i) for i in range(N)]
u = np.asarray([u1,u2])

# Do the same for the state and output variables
x1 = [z3.Real('x1_%s'%i) for i in range(N)]
x2 = [z3.Real('x2_%s'%i) for i in range(N)]
x3 = [z3.Real('x3_%s'%i) for i in range(N)]
x4 = [z3.Real('x4_%s'%i) for i in range(N)]
x = np.asarray([x1,x2,x3,x4])

y1 = [z3.Real('y1_%s'%i) for i in range(N)]
y2 = [z3.Real('y2_%s'%i) for i in range(N)]
y3 = [z3.Real('y3_%s'%i) for i in range(N)]
y4 = [z3.Real('y4_%s'%i) for i in range(N)]
y = np.asarray([y1,y2,y3,y4])

# Now we'll start adding up constraints 
constraints = []

# Initial conditions
for i in range(4):
    constraints.append(x[i,0]==x0[i,0])

# Dynamic constraints
for i in range(1,N):
    # unfortunately z3 doesn't really play well with matrix multiplication, so we have to do this by hand
    constraints.append(x1[i] == sys.A[0,0]*x1[i-1] + sys.A[0,1]*x2[i-1] + sys.A[0,2]*x3[i-1] + sys.A[0,3]*x4[i-1] + sys.B[0,0]*u1[i-1] + sys.B[0,1]*u2[i-1])
    constraints.append(x2[i] == sys.A[1,0]*x1[i-1] + sys.A[1,1]*x2[i-1] + sys.A[1,2]*x3[i-1] + sys.A[1,3]*x4[i-1] + sys.B[1,0]*u1[i-1] + sys.B[1,1]*u2[i-1])
    constraints.append(x3[i] == sys.A[2,0]*x1[i-1] + sys.A[2,1]*x2[i-1] + sys.A[2,2]*x3[i-1] + sys.A[2,3]*x4[i-1] + sys.B[2,0]*u1[i-1] + sys.B[2,1]*u2[i-1])
    constraints.append(x4[i] == sys.A[3,0]*x1[i-1] + sys.A[3,1]*x2[i-1] + sys.A[3,2]*x3[i-1] + sys.A[3,3]*x4[i-1] + sys.B[3,0]*u1[i-1] + sys.B[3,1]*u2[i-1])

# Output constraints
for i in range(N):
    constraints.append(y1[i] == x1[i])  # x position
    constraints.append(y2[i] == x3[i])  # y position
    constraints.append(y3[i] == u1[i])  # x acceleration input
    constraints.append(y4[i] == u2[i])  # y acceleration input

# Control constraints (maybe part of the specification?
umin = -1.0
umax = 1.0
for i in range(N):
    constraints.append(umin < u1[i])
    constraints.append(u1[i] < umax)
    constraints.append(umin < u2[i])
    constraints.append(u2[i] < umax)

constraints.append(u1[0] == 0.9)

# Specification constraints
spec = sys.control_bounded
constraints.append( mymax(u1[0],u1[1]+0.2) < 1.0)
#constraints.append( spec.robustness(y,0) > 0)
#print(sys.full_specification.robustness(y,0))



# Finally, use z3 to find evaluations of the state, input, and output variables that
# satisfy the constraints
s = z3.Solver()
s.add(constraints)
print(s.check())

if str(s.check()) == 'sat':
    m = s.model()

    # Extract the resulting trajectory
    x_trajectory = [real_to_float(m[i]) for i in x1]
    y_trajectory = [real_to_float(m[i]) for i in x3]
    u_trajectory = [real_to_float(m[i]) for i in u1]

    print(x_trajectory)
    print(u_trajectory)

    # Plot the results
    fig, ax = plt.subplots(1)
    sys.plot_scenario(ax)
    ax.plot(x_trajectory, y_trajectory, label="trajectory", linestyle="-", marker="o")

    plt.show()


