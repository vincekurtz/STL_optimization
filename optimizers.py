##
#
# This file contains custom optimization functions for use with
# scipy.minimize. 
#
# As noted in the documentation, each routine takes a function (func(x))
# and an initial guess x0, and returns an OptimizeResult object.
#
# See https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html#custom-minimizers
# for more details. 
#
##

import numpy as np
from scipy.optimize import OptimizeResult
from scipy.optimize import basinhopping, differential_evolution
from skopt import gp_minimize

def basinhopping_method(fun, x0, args=(), **options):
    """
    Basinhopping, a simulated annealing-like approach.
    This is a wrapper for scipy.optimize.basihopping.
    """
    res = basinhopping(fun, x0, disp=True)
    return res

def genetic_algorithm(fun, x0, args=(), **options):
    """
    Optimization via a Genetic Algorithm.
    This is a wrapper for scipy.optimize.differential_evolution
    """
    bounds = [(-0.9,0.9) for i in range(len(x0))]  # set limits of search space
    res = differential_evolution(fun, bounds,
                                        disp=True)
    return res

def gp_bayesian(fun, x0, args=(), **options):
    """
    Baysian optimization with Gaussian Processes
    This is a wrapper for skopt.gp_minimize.
    """
    dimensions = [(-0.9,0.9) for i in range(len(x0))]  # set limits of search space
    res = gp_minimize(fun, dimensions, 
                                verbose=True, 
                                acq_func="EI",
                                noise=1e-9)
    res.x = np.asarray(res.x)  # ensures output is a numpy array
    return res

def cross_entropy(fun, x0, args=(), **options):
    """
    A native implementation of cross entropy optimization.
    Thanks to Zhongjiao Shi for this implementation.
    """
    # number of iterations
    if 'niter' in options.keys():
        M = options['niter']
    else:
        M = 200

    T = int(len(x0)/2)

    pol_mean = np.zeros((2,T),dtype=float)    #Pre-alocate the policy mean
    pol_cov = np.ones((2,2,T),dtype=float)   #Pre-alocate the policy covariance

    last_pol_mean = np.zeros((2,T),dtype=float)
    last_pol_cov = np.zeros((2,2,T),dtype=float)

    #Given the initial policy
    for t in range(T):
        pol_cov[:,:,t] = 2*np.diag([1,1])

    for m in range(M):

        #Sample Size at iteration m
        N = int(np.amax([np.power(m,1.1),50]))

        #Elite group size
        K = int(0.02*N)

        #sampled control signal
        con = np.zeros((2,T,N),dtype=float)

        #elite group
        elite_con = np.zeros((2,T,K),dtype=float)

        #total robustness degree
        J = np.zeros((2,N),dtype=float)
        J[0,:] = range(N)

        last_pol_mean=pol_mean
        last_pol_cov=pol_cov

        for n in range(N):
            #Sample control inputs
            for t in range(T):
                con[:,t,n] = np.random.multivariate_normal(pol_mean[:,t],pol_cov[:,:,t])

            #Calculate total robustness degree
            inpt = con[:,:,n].flatten()   # flatten so that input to fun() is 1d
            J[1,n] = -fun(inpt)     # we'll use negative since this implementation assumes maximization

        # print verbose output if specified in options
        if ('disp' in options.keys()) and options['disp'] and (m%5==0):
            print("Iteration %s / %s   |   iteration average f(x) : %0.3f" % (m,M, -np.mean(J[1,:])))

        #sorting the robustness degree select the elite group
        J = J[:,J[1].argsort()]
        for k in range(K):
            elite_con[:,:,k] = con[:,:,int(J[0,N-1-k])]


        con_var = np.zeros((2,2,K),dtype=float)
        #Update the parameter
        alk = 2.0/(np.power(m+150,0.501))

        for t in range(T):
            pol_mean[:,t] = alk*np.mean(elite_con[:,t,:],axis=1)+(1-alk)*last_pol_mean[:,t]
            for k in range(K):
                con_var[:,:,k] = np.outer(elite_con[:,t,k]-pol_mean[:,t],elite_con[:,t,k]-pol_mean[:,t])+0.001*np.diag([1,1])

            pol_cov[:,:,t] = alk*np.mean(con_var,axis=2)+(1-alk)*(last_pol_cov[:,:,t]+np.outer(last_pol_mean[:,t]-pol_mean[:,t],last_pol_mean[:,t]-pol_mean[:,t]))


    # Get the optimal policy
    x_opt = np.zeros((2,T),dtype=float)
    for t in range(T):
        x_opt[:,t] = np.random.multivariate_normal(pol_mean[:,t],pol_cov[:,:,t])

    res = OptimizeResult
    res.x = x_opt
    res.success = True
    res.nit = M

    return res