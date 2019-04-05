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


# General Heteroscedastic BO Stuff -----------------------------------
def LCB(model, X, beta=1):
    """
    Computes the lower confidence bound of the prediction model `model`
    at position X.
    """
    mu, sigma = model.predict_noiseless(X[:,None])
    lcb = mu - beta*sigma
    return lcb

def get_next_sample(model, X, bounds, n_restarts=25):
    """
    Use the LCB to find the next sample.
    """
    dim = X.shape[1]
    min_val = 1e5  # something big to start
    min_x = None

    def min_obj(X):
        return LCB(model, X, beta=5.0)

    for x0 in np.random.uniform(bounds[:,0], bounds[:,1], size=(n_restarts, dim)):
        res = minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B')
        if res.fun[0] < min_val:
            min_val = res.fun[0]
            min_x = res.x

    return min_x.reshape(-1,dim)

def bayes_opt(f, var_f, bounds, n_iter=10):

    # Take initial samples and record the associated uncertainty
    X = np.linspace(-1,2,num=7)[:,None]
    error = np.random.multivariate_normal(np.zeros(X.size),np.diag(var_f(X).flatten()))[:,None]
    Y = f(X) + error

    for i in range(n_iter):
        # Update the GP with existing samples
        kern = GPy.kern.RBF(input_dim=1,variance=1.5,lengthscale=0.5)# + GPy.kern.Bias(1)
        m = GPy.models.GPHeteroscedasticRegression(X,Y,kern)

        # Use the (known) uncertainty on each observation
        m['.*het_Gauss.variance'] = var_f(X) #Set the noise parameters to the error in Y
        m.het_Gauss.variance.fix() #We can fix the noise term, since we already know it
        #m.optimize()

        # Get the next sampling point
        X_next = get_next_sample(m,X,bounds)

        # Get the next sample and uncertainty estimate
        error_next = np.random.normal(0,var_f(X_next))
        Y_next = f(X_next) + error_next

        # Append these new values to the data
        X = np.vstack([X_next,X])
        Y = np.vstack([Y_next,Y])
        error = np.vstack([error_next,error])

    return m, X, Y, error


if __name__=="__main__":
    # Target function that we'll approximate
    def f(X):
        return np.sin(3*X) + X**2 - 0.7*X

    # Gives the variance associated with each evaluation of f(X)
    def uncertainty_fcn(X):
        return abs(0.4 - 0.3*X)

    # Bounds on the input space
    bounds = np.asarray([[-1.0,2.0]])
    
    # Run the optimization
    m, X, Y, error = bayes_opt(f, uncertainty_fcn, bounds, n_iter=50)

    print(m)
    print(uncertainty_fcn(-0.4))
    print(m.predict_noiseless(np.asarray([[-0.4]])))

    # Plot stuff
    m.plot_f(density=True)
    plt.plot(np.linspace(-1,2),f(np.linspace(-1,2)),'r-',label="True f(x)")
    plt.plot(X,Y,'kx',mew=1.5)
    plt.errorbar(X,Y,yerr=abs(error.flatten()),fmt="none",ecolor='k',zorder=1)

    plt.xlim(bounds.flatten())
    plt.legend()

    plt.show()

