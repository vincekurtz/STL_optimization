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

import jax.numpy as np
from scipy.optimize import OptimizeResult
from scipy.optimize import basinhopping, differential_evolution
from skopt import gp_minimize
from multiprocessing import Process, Manager

def basinhopping_method(fun, x0, args=(), **options):
    """
    Basinhopping, a simulated annealing-like approach.
    This is a wrapper for scipy.optimize.basihopping.
    """
    # set termination condition: exit if robustness is above epsilon
    epsilon = 0.05
    callback = lambda x, f, accept : True if fun(x) <= -epsilon else False
    res = basinhopping(fun, x0, disp=True, callback=callback)
    return res

def differential_evolution_method(fun, x0, args=(), **options):
    """
    Optimization via Differential Evolution
    This is a wrapper for scipy.optimize.differential_evolution
    """
    bounds = [(-0.9,0.9) for i in range(len(x0))]  # set limits of search space

    # sets termination condition: exit if the robustness is above epsilon
    epsilon = 0.1
    callback = lambda xk, convergence : True if fun(xk) <= -epsilon else False

    res = differential_evolution(fun, bounds,
                                        callback = callback,
                                        maxiter = 5,
                                        disp=True)
    return res

def gp_bayesian(fun, x0, args=(), **options):
    """
    Baysian optimization with Gaussian Processes
    This is a wrapper for skopt.gp_minimize.
    """
    dimensions = [(-0.9,0.9) for i in range(len(x0))]  # set limits of search space
   
    # set termination condition: exit if the robustness is above epsilon
    epsilon = 0.001
    callback = lambda res : True if res.fun <= -epsilon else False

    res = gp_minimize(fun, dimensions, 
                                verbose=True, 
                                n_calls=100,
                                acq_func="EI",
                                callback=callback,
                                n_jobs=-1,
                                noise=1e-9)
    res.x = np.asarray(res.x)  # ensures output is a numpy array
    return res

def fast_bayesian(fun, x0, args=(), **options):
    """
    Baysian optimization with differential evolution initialization. 
    The idea is to be fast and probabilistically complete. 
    """

    # bounds for the search space
    bounds = [(-0.9, 0.9) for i in range(len(x0))]

    # number of times to search with differential evolution, and number of iterations on each restart
    de_restarts = 8
    de_iter = 5

    x_guess = []  # initial guesses to use for Bayesian opt. 


    print("\n===> Finding initial point with Differential Evolution\n")

    # We'll use multithreading!
    # calculate an initial guess and add it to a list
    def target_func(lst, i):
        np.random.seed()   # need to ensure each thread gets a different random start
        res = differential_evolution(fun, bounds, disp=True, maxiter=de_iter)
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
            x_guess.append(L[i])

    # Termination condition for bayesian opt: exit if the robustness is above this level
    epsilon = 0.001
    callback = lambda res : True if res.fun <= -epsilon else False

    # Now do bayesian optimization until we reach the threshold
    print("\n===> Optimizing with Bayesian Optimization\n")
    res = gp_minimize(fun, bounds,
                            verbose=True,
                            n_calls=200,
                            acq_func="LCB",   # use Lower Confidence Bound for better theoretical gaurantees
                            kappa=1.01,
                            callback=callback,
                            n_jobs=-1,
                            noise=1e-9,
                            x0=x_guess)

    # ensure output is a np array
    res.x = np.asarray(res.x)
    return res
                                                

def cross_entropy(fun, x0, args=(), **options):
    """
    A native implementation of cross entropy optimization.
    Thanks to Zhongjiao Shi for this implementation.
    """
    # set termination condition: exit if the robustness is above epsilon
    epsilon = 0.0

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

        # exit if the average prediction is better than the threshold
        if np.mean(J[1,:]) > epsilon:
            print("Done!")
            break

    # Get the optimal policy
    x_opt = np.zeros((2,T),dtype=float)
    for t in range(T):
        x_opt[:,t] = np.random.multivariate_normal(pol_mean[:,t],pol_cov[:,:,t])

    res = OptimizeResult
    res.x = x_opt
    res.success = True
    res.nit = M

    return res
