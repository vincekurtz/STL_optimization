#!/usr/bin/env python2

##
#
# An example of using heteroscedastic (different observation noises) GP regression 
# for Bayesian Optimization
#
##

import numpy as np
import matplotlib.pyplot as plt
import GPy

# Target function that we'll approximate
def f(X):
    return 10. + .07*X + 2*np.sin(X)/X

# Drawing noisy samples from the target
Ns = 10
X = np.random.uniform(-10,20,Ns)
error = np.random.normal(0,0.2,X.size)
Y = f(X) + error

for i in range(1,Ns):

    # Only consider a subset of the samples
    Xi = X[0:i]
    Yi = Y[0:i]
    error_i = error[0:i]

    # Regression
    kern = GPy.kern.RBF(input_dim=1,variance=1,lengthscale=3) + GPy.kern.Bias(1)

    # Assume we know the errors on each observation
    m = GPy.models.GPHeteroscedasticRegression(Xi[:,None],Yi[:,None],kern)
    m['.*het_Gauss.variance'] = abs(error_i)[:,None] #Set the noise parameters to the error in Y
    m.het_Gauss.variance.fix() #We can fix the noise term, since we already know it
    m.optimize()

    m.plot_f() #Show the predictive values of the GP.
    plt.plot(Xi,Yi,'kx',mew=1.5)
    plt.errorbar(Xi,Yi,yerr=np.array(m.likelihood.flattened_parameters).flatten(),fmt="none",ecolor='k',zorder=1)
    plt.plot(np.linspace(-10,20),f(np.linspace(-10,20)),'r-',label="True f(x)")
    plt.ylim((8,13))

    plt.legend()

    plt.show()
