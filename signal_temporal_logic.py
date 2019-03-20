##
#
# Code for specifing STL formulas, evaluating the robustness
# of signals, and specifying robustness constraints in z3
#
##

import numpy as np
import z3

# Helper functions for z3 min and max, since z3 only can handle basic
# math operators from python, along with its own built-in logical functions
def z3min(a,b):
    return z3.If(a < b, a, b)

def z3max(a,b):
    return z3.If(a > b, a, b)

def z3listmin(l):
    my_min = l[0]
    for i in range(1,len(l)):
        my_min = z3min(my_min, l[i])
    return my_min

def z3listmax(l):
    my_max = l[0]
    for i in range(1,len(l)):
        my_max = z3max(my_max, l[i])
    return my_max

# Main STL formula class definition
class STLFormula:
    """
    An STL Formula. These can be built up from predicates using logical
    operations specified in this class. 
    """
    def __init__(self, robustness, z3_robustness=None):
        """
        An STL Formula is initialized with a robustness function, which
        is commonly specified using a Predicate. 

        More complex formulas are determined by logical operations
        like conjuction and disjunction (see other methods in this class).

        Arguments:
            robustness : a function that maps from signal s and time t to a scalar value
            z3_robustness : the robustness function using only basic math operators and z3 operators
        """
        self.robustness = robustness

        if z3_robustness is None:
            self.z3_robustness = robustness
        else:
            self.z3_robustness = z3_robustness

    def negation(self):
        """
        Return a new STL Formula object which represents the negation
        of this one. The robustness degree is given by

            rho(s,-phi,t) = -rho(s,phi,t)
        """
        new_robustness = lambda s, t : - self.robustness(s,t)
       
        new_z3_robustness = lambda s, t : - self.z3_robustness(s,t)
        
        new_formula = STLFormula(new_robustness, new_z3_robustness)

        return new_formula

    def conjunction(self, second_formula):
        """
        Return a new STL Formula object which represents the conjuction of
        this formula with second_formula:

            rho(s,phi1^phi2,t) = min( rho(s,phi1,t), rho(s,phi2,t) )

        Arguments:
            second_formula : an STL Formula or predicate defined over the same signal s.
        """
        new_robustness = lambda s, t : min( self.robustness(s,t),
                                            second_formula.robustness(s,t) )

        new_z3_robustness = lambda s, t : z3min( self.z3_robustness(s,t),
                                                 second_formula.z3_robustness(s,t) )

        new_formula = STLFormula(new_robustness, new_z3_robustness)

        return new_formula

    def disjunction(self, second_formula):
        """
        Return a new STL Formula object which represents the disjunction of
        this formula with second_formula:

            rho(s, phi1 | phi2, t) = max( rho(s,phi1,t), rho(s,phi2,t) )

        Arguments:
            second_formula : an STL Formula or predicate defined over the same signal s.
        """
        new_robustness = lambda s, t : max( self.robustness(s,t),
                                            second_formula.robustness(s,t) )
        
        new_z3_robustness = lambda s, t : z3max( self.z3_robustness(s,t),
                                                 second_formula.z3_robustness(s,t) )

        new_formula = STLFormula(new_robustness, new_z3_robustness)

        return new_formula

    def eventually(self, t1, t2):
        """
        Return a new STL Formula object which represents this formula holding
        at some point in [t+t1, t+t2].

            rho(s, F_[t1,t2](phi), t) = max_{k in [t+t1,t+t2]}( rho(s,phi,k) )

        Arguments:
            t1 : an integer between 0 and signal length T
            t2 : an integer between t1 and signal length T
        """
        new_robustness = lambda s, t : max([ self.robustness(s,k) for k in range(t+t1, t+t2+1)])
        
        new_z3_robustness = lambda s, t : z3listmax([ self.z3_robustness(s,k) for k in range(t+t1, t+t2+1)])

        new_formula = STLFormula(new_robustness, new_z3_robustness)

        return new_formula

    def always(self, t1, t2):
        """
        Return a new STL Formula object which represents this formula holding
        at all times in [t+t1, t+t2].

            rho(s, F_[t1,t2](phi), t) = min_{k in [t+t1,t+t2]}( rho(s,phi,k) )

        Arguments:
            t1 : an integer between 0 and signal length T
            t2 : an integer between t1 and signal length T
        """
        new_robustness = lambda s, t : min([ self.robustness(s,k) for k in range(t+t1, t+t2+1)])

        new_z3_robustness = lambda s, t : z3listmin([ self.z3_robustness(s,k) for k in range(t+t1, t+t2+1)])

        new_formula = STLFormula(new_robustness, new_z3_robustness)

        return new_formula

