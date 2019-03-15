##
#
# This file contains classes that define some simple examples of scenarios
# where a task is specified using an STL formula.
#
##

from copy import copy
import numpy as np
from pySTL import STLFormula
from matplotlib.patches import Rectangle

class ReachAvoid:
    """
    This example involves moving a robot with double integrator
    dynamics past an obstacle and to a goal postion with bounded
    control effort. 

    It also serves as a template class for more complex examples.
    """
    def __init__(self, initial_state):
        """
        Set up the example scenario with the initial state, which should be a (4,1) numpy
        array with [x,x',y,y'].
        """

        self.x0 = np.asarray(initial_state)

        self.T = 20  # The time bound of our specification

        # Obstacle and goal region vertices: (xmin, xmax, ymin, ymax)
        self.obstacle_vert = (3,5,4,6)
        self.goal_vert = (7,8,8,9)

        # Now we'll define the STL specification. We'll do this over
        # the signal s, which is a list of x, y coordinates and the control
        # input u at each timestep. 

        # Obstacle and goal constraints
        hit_obstacle = self.in_rectangle_formula(self.obstacle_vert) 
        at_goal = self.in_rectangle_formula(self.goal_vert)
        
        self.obstacle_avoidance = hit_obstacle.negation().always(0,self.T)
        self.reach_goal = at_goal.eventually(0,self.T)

        # Control constraints
        umin = - 0.2
        umax = 0.2
        u1_above_min = STLFormula(lambda s, t : s[t,2] - umin)
        u1_below_max = STLFormula(lambda s, t : -s[t,2] + umax)
        u2_above_min = STLFormula(lambda s, t : s[t,3] - umin)
        u2_below_max = STLFormula(lambda s, t : -s[t,3] + umax)

        u1_valid = u1_above_min.conjunction(u1_below_max)
        u2_valid = u2_above_min.conjunction(u2_below_max)

        self.control_bounded = u1_valid.conjunction(u2_valid).always(0,self.T)

        # Full specification
        self.full_specification = self.obstacle_avoidance.conjunction(self.reach_goal).conjunction(self.control_bounded)

    def in_rectangle_formula(self, rectangle):
        """
        A helper function which returns an STL Formula denoting that 
        the x,y position (first two elements of signal s) is in the given rectangle.
        """
        # unpack vertices
        xmin, xmax, ymin, ymax = rectangle

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

    def STL_signal(self, u):
        """ 
        Maps a control signal u and an initial condition to an STL signal we can check. 
        This signal we will check is composed of the (x,y) position of the robot 
        and the control inputs.

        Arguments:
            u   : a (2,T) numpy array representing the control sequence

        Returns:
            s   : a (4,T) numpy array representing the signal we'll check
        """
        # System definition: x_{t+1} = A*x_t + B*u_t
        A = np.array([[1,1,0,0],[0,1,0,0],[0,0,1,1],[0,0,0,1]])
        B = np.array([[0,0],[1,0],[0,0],[0,1]])

        T = u.shape[1]      # number of timesteps

        # Pre-alocate the signal
        s = np.zeros((4,T)) 

        # Run the controls through the system and see what we get
        x = copy(self.x0)
        for t in range(T):
            # extract the first and third elements of x
            s[0:2,t] = x[[0,2],:].flatten()   
            s[2:4,t] = u[:,t]

            # Update the system state
            x = A@x + B@u[:,t][:,np.newaxis]   # ensure u is of shape (2,1) before applying

        return s

    def rho(self, u, spec=None):
        """
        For a given initial state and control sequence u, calculates rho,
        a scalar value which indicates the degree of satisfaction of the specification.

        Arguments:
            u    : a (2,T) numpy array representing the control sequence
            spec : an STLFormula to evaluate (the full specification by default)

        Returns:
            rho  : a scalar value indicating the degree of satisfaction. Positive values
                     indicate that the specification is satisfied.
        """
        # By default, evaluate the full specification. Otherwise you could pass it 
        # a different formula, such as the (sub)specification for obstacle avoidance.
        if spec is None:
            spec = self.full_specification

        s = self.STL_signal(u)
        rho = spec.robustness(s.T, 0)

        return rho

    def cost_function(self, u):
        """
        Defines a cost function over the control sequence u such that
        the optimal u maximizes the robustness degree of the specification.

        Arguments:
            u    : a (m*T,) flattened numpy array representing a tape of control inputs

        Returns:
            J    : a scalar value indicating the degree of satisfaction of the specification.
                   (negative ==> satisfied)
        """
        # enforce that the input is a numpy array
        u = np.asarray(u)

        # Reshape the control input to (mxT). Vector input is required for some optimization libraries
        T = int(len(u)/2)
        u = u.reshape((2,T))

        J = - self.rho(u)

        return J

    def plot_scenario(self, ax):
        """
        Create a plot of the obstacle and goal regions on
        the given matplotlib axis.
        """
        ax.set_xlim((0,12))
        ax.set_ylim((0,12))

        # Unpack region's sizes and positions
        obs_x = self.obstacle_vert[0]
        obs_y = self.obstacle_vert[2]
        obs_w = self.obstacle_vert[1]-obs_x
        obs_h = self.obstacle_vert[3]-obs_y

        tar_x = self.goal_vert[0]
        tar_y = self.goal_vert[2]
        tar_w = self.goal_vert[1]-tar_x
        tar_h = self.goal_vert[3]-tar_y

        obstacle = Rectangle((obs_x,obs_y),obs_w,obs_h,color='red',alpha=0.5)
        target = Rectangle((tar_x,tar_y),tar_w,tar_h, color='green',alpha=0.5)

        ax.add_patch(obstacle)
        ax.add_patch(target)

    def plot_trajectory(self, u, ax, label=None):
        """
        Create a plot of the position resulting from applying 
        control u on the matplotlib axis ax. 
        """
        # Plot the goal and obstacle regions
        self.plot_scenario(ax)

        # Get the resulting trajectory
        s = self.STL_signal(u)
        x_position = s[0,:]
        y_position = s[1,:]

        ax.plot(x_position,y_position, label=label, linestyle="-", marker="o")

    def print_trajectory(self, u):
        """
        Print out the resulting trajectory in a way that can it can be copied
        to another python program for postprocessing/pretty formatting.
        """
        # Get the trajectory
        s = self.STL_signal(u)
        trajectory = s[0:2,:]   # x and y position

        print(repr(trajectory))

class EitherOr(ReachAvoid):
    """
    This example involves moving a robot with double integrator
    dynamics past an obstacle and to a goal postion with bounded
    control effort, but first reaching one of two target regions
    """
    def __init__(self, initial_state, T=20):
        """
        Set up the example scenario with the initial state, which should be a (4,1) numpy
        array with [x,x',y,y'].
        """

        self.x0 = np.asarray(initial_state)

        self.T = T  # The time bound of our specification

        # Obstacle and goal region vertices: (xmin, xmax, ymin, ymax)
        self.obstacle_vert = (3,5,4,6)
        self.goal_vert = (7,8,8,9)
        self.target1_vert = (6,7,4.5,5.5)
        self.target2_vert = (1,2,4.5,5.5)

        # Now we'll define the STL specification. We'll do this over
        # the signal s, which is a list of x, y coordinates and the control
        # input u at each timestep. 

        # Obstacle and goal constraints
        hit_obstacle = self.in_rectangle_formula(self.obstacle_vert) 
        at_goal = self.in_rectangle_formula(self.goal_vert)
        
        self.obstacle_avoidance = hit_obstacle.negation().always(0,self.T)
        self.reach_goal = at_goal.eventually(0,self.T)

        # Intermediate target constraints
        at_target1 = self.in_rectangle_formula(self.target1_vert)
        reach_target1 = at_target1.eventually(0,self.T)
        
        at_target2 = self.in_rectangle_formula(self.target2_vert)
        reach_target2 = at_target2.eventually(0,self.T)

        self.intermediate_target = reach_target1.disjunction(reach_target2)

        # Control constraints
        umin = - 0.9
        umax = 0.9
        u1_above_min = STLFormula(lambda s, t : s[t,2] - umin)
        u1_below_max = STLFormula(lambda s, t : -s[t,2] + umax)
        u2_above_min = STLFormula(lambda s, t : s[t,3] - umin)
        u2_below_max = STLFormula(lambda s, t : -s[t,3] + umax)

        u1_valid = u1_above_min.conjunction(u1_below_max)
        u2_valid = u2_above_min.conjunction(u2_below_max)

        self.control_bounded = u1_valid.conjunction(u2_valid).always(0,self.T)

        # Full specification
        self.full_specification = self.obstacle_avoidance.conjunction(self.reach_goal).conjunction(self.control_bounded).conjunction(self.intermediate_target)

    def plot_scenario(self, ax):
        """
        Create a plot of the obstacle and goal regions on
        the given matplotlib axis.
        """
        ax.set_xlim((0,10))
        ax.set_ylim((0,10))

        # Unpack region's sizes and positions
        obs_x = self.obstacle_vert[0]
        obs_y = self.obstacle_vert[2]
        obs_w = self.obstacle_vert[1]-obs_x
        obs_h = self.obstacle_vert[3]-obs_y

        goal_x = self.goal_vert[0]
        goal_y = self.goal_vert[2]
        goal_w = self.goal_vert[1]-goal_x
        goal_h = self.goal_vert[3]-goal_y

        target1_x = self.target1_vert[0]
        target1_y = self.target1_vert[2]
        target1_w = self.target1_vert[1]-target1_x
        target1_h = self.target1_vert[3]-target1_y

        target2_x = self.target2_vert[0]
        target2_y = self.target2_vert[2]
        target2_w = self.target2_vert[1]-target2_x
        target2_h = self.target2_vert[3]-target2_y

        obstacle = Rectangle((obs_x,obs_y),obs_w,obs_h,color='red',alpha=0.5)
        goal = Rectangle((goal_x,goal_y),goal_w,goal_h, color='green',alpha=0.5)

        target1 = Rectangle((target1_x,target1_y),target1_w,target1_h, color='blue',alpha=0.5)
        target2 = Rectangle((target2_x,target2_y),target2_w,target2_h, color='blue',alpha=0.5)

        ax.add_patch(obstacle)
        ax.add_patch(goal)
        ax.add_patch(target1)
        ax.add_patch(target2)

