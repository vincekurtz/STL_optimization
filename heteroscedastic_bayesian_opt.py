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
import GPyOpt
from scipy.optimize import minimize

def LCB(model, X, beta=1):
    """
    Computes the lower confidence bound of the prediction model `model`
    at position X.
    """
    mu, sigma = model.predict_noiseless(X[:,None])
    lcb = mu + beta*sigma
    return lcb

def get_next_sample(model, X, bounds, n_restarts=25):
    """
    Use the LCB to find the next sample.
    """
    dim = X.shape[1]
    min_val = 1e5  # something big to start
    min_x = None

    def min_obj(X):
        return LCB(model, X, beta=1.01)

    for x0 in np.random.uniform(bounds[:,0], bounds[:,1], size=(n_restarts, dim)):
        res = minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B')
        if res.fun[0] < min_val:
            min_val = res.fun[0]
            min_x = res.x

    return min_x.reshape(-1,dim)

# Target function that we'll approximate
def f(X):
    return -np.sin(3*X) - X**2 + 0.7*X

bounds = np.asarray([[-1.0,2.0]])

# Take initial samples and record the associated uncertainty
X = np.random.uniform(-1,2,5)[:,None]
error = np.random.normal(0,0.2,X.size)[:,None]
Y = f(X) + error

# Set up some plots
fig, ax = plt.subplots(1,2)

n_iter = 10
for i in range(n_iter):
    # Update the GP with existing samples
    kern = GPy.kern.RBF(input_dim=1,variance=1,lengthscale=3) + GPy.kern.Bias(1)
    m = GPy.models.GPHeteroscedasticRegression(X,Y,kern)

    # Use the (known) uncertainty on each observation
    m['.*het_Gauss.variance'] = abs(error) #Set the noise parameters to the error in Y
    m.het_Gauss.variance.fix() #We can fix the noise term, since we already know it
    m.optimize()

    # Get the next sampling point
    #X_next = np.asarray([np.random.uniform(-1,2)])
    X_next = get_next_sample(m,X,bounds)

    # Get the next sample and uncertainty estimate
    error_next = np.random.normal(0,0.2,1)
    Y_next = f(X_next) + error_next

    # Plot the estimate so far, along with the next point we'll sample
    ax[0].clear()
    m.plot_f(ax=ax[0])
    ax[0].plot(X,Y,'kx',mew=1.5)
    ax[0].errorbar(X,Y,yerr=np.array(m.likelihood.flattened_parameters).flatten(),fmt="none",ecolor='k',zorder=1)
    ax[0].plot(np.linspace(-1,2),f(np.linspace(-1,2)),'r-',label="True f(x)")
    ax[0].axvline(x=X_next, label="Next Sample", color='k', linestyle='--')
    ax[0].set_ylim((-2,2))
    ax[0].set_xlim((-1,2))
    ax[0].legend()

    ax[1].clear()
    ax[1].plot(np.linspace(-1,2),LCB(m,np.linspace(-1,2)),'r-',label="Aquisition Fcn")
    ax[1].axvline(x=X_next, label="Next Sample", color='k', linestyle='--')
    ax[1].legend()

    plt.draw()
    plt.pause(0.01)

    raw_input("Press [enter] to continue ")


    # Append these new values to the data
    X = np.vstack([X_next,X])
    Y = np.vstack([Y_next,Y])
    error = np.vstack([error_next,error])

plt.show()
