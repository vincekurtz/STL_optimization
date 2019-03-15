#!/usr/bin/env python3

##
#
# This is a quick script to plot the trajectories resulting from different
# methods on the same plot. We can also do this with comparison.py, but this
# way allows us to compare directly with MILP, which we don't implement here.
#
##

import numpy as np
import matplotlib.pyplot as plt
from example_scenarios import EitherOr

# Bayesian and Differential Evolution (ours)
bayes_de = np.array([[  0.        ,   0.        ,   0.40589072,   1.07705817,
          1.25753808,   1.59868652,   2.3810179 ,   3.7391193 ,
          5.58559533,   7.51505297,   8.88022136,  10.57330406,
         12.33943439,  14.22233127,  15.53800547,  16.39248662,
         17.70622925,  19.65664343,  22.07932351,  23.87928503,
         25.46532034,  26.83402606,  27.81499878,  28.70118186,
         29.18346343,  29.73046001],
       [  0.        ,   0.        ,   0.46280145,   1.51908195,
          2.94267607,   4.74869325,   6.00299569,   7.41522627,
          8.1761501 ,   8.53863688,   8.58243583,   9.19767658,
          9.58281266,   9.63684499,   9.24042164,   8.38195576,
          7.23940624,   5.97687117,   4.09515693,   1.87119897,
         -0.42221082,  -2.61888229,  -4.70272269,  -7.02074232,
         -9.07477992, -11.57816735]])
bayes_only = np.array([[ 0.        ,  0.        ,  0.9       ,  2.7       ,  4.17876169,
         6.05964957,  7.8616493 , 10.56364903, 14.04838544, 16.63312185,
        19.56277241, 22.4394923 , 25.72151104, 28.94476359, 33.00676685,
        36.78232368, 40.44508108, 43.62579294, 46.7455863 , 49.78947878,
        51.93337125, 53.9746811 , 55.11599095, 56.86829866, 58.23722201,
        60.50614535],
       [ 0.        ,  0.        ,  0.9       ,  2.15018693,  3.52511296,
         5.55260607,  8.48009918, 10.80055611, 13.1397553 , 16.37895449,
        19.40945241, 22.78858484, 26.01064856, 28.77929896, 30.83728748,
        33.79527601, 36.13239412, 38.82021177, 41.97267112, 44.22513047,
        47.23110768, 50.95395504, 54.47768564, 57.91981577, 60.87626646,
        63.71243011]])
DE_trajectory = np.array([[ 0.        ,  0.        , -0.28030078, -0.27876673, -0.5738335 ,
        -1.12219717, -0.86663345, -0.94453174, -0.74063096, -0.55484271,
         0.34165481,  1.33208709,  2.52353666,  3.63865814,  4.23842935,
         4.68520265,  4.75485459,  5.20456254,  5.51719713,  5.80676006,
         5.44802709,  5.66453646,  6.02449398,  6.24696097,  7.09638449,
         7.73855101],
       [ 0.        ,  0.        ,  0.62252809,  1.36556185,  2.89870913,
         3.82920259,  4.43967407,  4.62210637,  4.09256274,  4.09906661,
         4.47504123,  4.81214299,  5.56976453,  6.93938765,  8.01662249,
         8.70636162,  9.50616682, 10.28208942, 10.38454851, 11.07785506,
        10.98220712, 10.32404833,  9.81854771,  9.0117156 ,  8.16070681,
         6.98165327]])
milp = np.array([[ 0.00000000e+00,  0.00000000e+00,  1.07417181e-02,
         4.06969249e-02,  9.83373907e-02,  1.92134886e-01,
         3.30561182e-01,  5.22088048e-01,  7.75187256e-01,
         1.09833058e+00,  1.49998978e+00,  1.98863663e+00,
         2.54826151e+00,  3.16285479e+00,  3.81640684e+00,
         4.49290804e+00,  5.17634875e+00,  5.85071936e+00,
         6.50001023e+00,  7.10821174e+00,  7.49998978e+00,
         7.50001023e+00,  7.49998978e+00,  7.49999386e+00,
         7.50001022e+00],
       [ 0.00000000e+00,  0.00000000e+00, -1.43905493e-02,
        -3.43459566e-02, -5.10405304e-02, -5.56485796e-02,
        -3.93444126e-02,  6.69766177e-03,  9.13033349e-02,
         2.23298298e-01,  4.11508243e-01,  6.64758861e-01,
         9.91875843e-01,  1.40168488e+00,  1.90301166e+00,
         2.50468189e+00,  3.21552124e+00,  4.04435541e+00,
         5.00001010e+00,  6.09131099e+00,  7.27304268e+00,
         8.49998977e+00,  9.72693686e+00,  1.09538840e+01,
         1.21808310e+01]])

# Set up the scenario for plotting
x0 = np.asarray([0,0,0,0])[:,np.newaxis]
sys = EitherOr(x0)

# Default cycle of colors so we can match other plots
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

plt.figure()
# Bayesian and Differential Evolution (ours)
time = 20.6
rho = 0.248
sys.plot_scenario(plt.gca())
plt.plot(bayes_de[0,:],bayes_de[1,:],marker="x",
                                    color=colors[0],
                                    label="Compute Time: %ss  Robustness: %s" % (time, rho)
        )
plt.xlim([-2,10])
plt.ylim([-1,12])
plt.legend()

plt.figure()
# Bayesian Only
time = 136.1
rho = -0.05
sys.plot_scenario(plt.gca())
plt.plot(bayes_only[0,:],bayes_only[1,:],marker="o",
                                         color=colors[1],
                                         label="Compute Time: %ss  Robustness: %s" % (time, rho)
                                         )
plt.xlim([-2,10])
plt.ylim([-1,12])
plt.legend()

plt.figure()
# Differential Evolution Only
time = 8.80
rho = 0.096
sys.plot_scenario(plt.gca())
plt.plot(DE_trajectory[0,:],DE_trajectory[1,:],marker="^",
                                               color=colors[2],
                                               label="Compute Time: %ss  Robustness: %s" % (time, rho)
                                              )
plt.xlim([-2,10])
plt.ylim([-1,12])
plt.legend()

plt.figure()
# MILP
time = 74.839
rho = 0.5
sys.plot_scenario(plt.gca())
plt.plot(milp[0,:],milp[1,:],marker="s",
                             color=colors[3],
                             label="Compute Time: %ss  Robustness: %s" % (time, rho)
        )
plt.xlim([-2,10])
plt.ylim([-1,12])
plt.legend()

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
print(colors)

plt.show()
