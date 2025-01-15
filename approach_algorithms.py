#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fourier
"""

import autograd.numpy as np  # Thinly-wrapped numpy
import autograd as ag
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import cvxpy as cp

# Run random_quadratic_form.ipynb to obtain random generator functions
# to test the algorith with weakly convex functions

with open("random_quadratic_form.py") as f:
    exec(f.read())

# Function $\nabla F_k(y) = y-x + \frac{1}{\gamma_k}Dh(y)^{*}( [h(y)]_{+})$

# Function [x]_{+} 
def positive_part(x):
    """ 
    **Computes the positive part of the input array x.**
    
    Parameters:
    - x (array): Input array.

    Returns:
    - array: Output array with positive parts of array x. 
    """
    return np.maximum(0, x)

### Approach 1

# Function gradF_k
def gradF_k(y ,x , h, gamma_k):
    Dh = ag.jacobian(h)
    return y - x + (1.0 / gamma_k) * (Dh(y).T @ positive_part(h(y)))

# Algorithm: Approach 1 for C = R_{-}^m

def AlgorithmApproach1(x, y_0, h, gamma, lamb, eps):
    """ 
    **Perform Approach 1 for solving the optimization problem C = R_{-}^m.**

    Parameters:
    - x (array): Input data for the optimization problem, we expect x to be projected onto h^{-1}(C).
    - y_0 (array): Initial guess for the optimization variable.
    - h (function): Function to evaluate the optimization variable h(y) \in C, must have input an n-dimensional numpy array with output an m-dimensional numpy array.
    - gamma (function): Function to calculate the gamma parameter.
    - lambda (function): Function to calculate the lambda parameter.
    - eps (float): Tolerance for stopping the optimization.

    Returns:
    - y_optimal (array): Optimized variable that minimizes the objective function.
    - y_values (list): Sequence of optimized variables throughout the iterations.
    """
    
    # Initial iteration
    k = 0
    convergenceCondition = True
    y_k_plus = y_0 - lamb(k)*gradF_k(y_0, x, h, gamma(k))
    
    # List to construct sequence (y_n)
    y_values = [y_0]
    y_values.append(y_k_plus)
    
    # Start iterating the algorithm
    while convergenceCondition: 
        # Update the iteration counter
        k += 1
        
        # Update y_{k} = y_{k+1} and \gamma_k in reach iteration
        y_k = y_k_plus
        
        # Compute y_{k+1} = y_{k} - \lambda_{k} \grad F_{k}
        y_k_plus = y_k -  lamb(k) * gradF_k(y_k, x, h, gamma(k))
        
        # Save it in the sequence (y_values)
        y_values.append(y_k_plus)
        
        #Print to test algorithm
        print(y_k_plus)
        
        # Check the convergence condition ||y_{k+1} - y_{k}|| < eps to stop the optimization
        if np.linalg.norm(y_k_plus - y_k) < eps:
            convergenceCondition = False
            y_optimal = y_k_plus
            print('Solution Obtained: y = ', y_optimal, '| Evaluation in h(y) =', h(y_optimal))
            print('Number of iterations necessary to satisfy the condition ||y_{k+1}-y_{k}|| <', eps, 'are k =', k)
        
        elif np.linalg.norm(y_k_plus - y_k) > 10**10:
            y_optimal = y_k_plus
            print("Error: Algorithm diverges to infty")
            break

    return y_optimal, y_values

### Approach 2

# Function H(y;y_k;h)
def H(y,y_k,h):
    """ 
    **Computes the affine function H(y, y_k) given the function h.**
    Parameters:
    - y (array): Optimization variable.
    - y_k (array): Previous optimization variable.
    - h (function): Function to evaluate h(y), must have as input an n-dimensional numpy array with output an m-dimensional numpy array.

    Returns:
    - array: Result of the affine function H(y, y_k). 
    """
    
    # Define the diferential Dh(\cdot) using autograd Jacobian approximation of the given h(x) function
    Dh = ag.jacobian(h)
    return h(y_k) + Dh(y_k) @ (y-y_k)

# Example of objective Function F_{k}(y)
def F_k_module(y,x,h,y_k,gamma_k,beta):
    """ 
    **Computes the objective function F_k(y) for a given iteration k.**

    Parameters:
    - y (array): Optimization variable.
    - x (array): Input data for the optimization problem.
    - h (function): Function to evaluate h(y).
    - y_k (array): Previous optimization variable.
    - gamma_k (float): Current gamma parameter.
    - beta (float): Parameter for the optimization problem.

    Returns:
    - float: Result of the objective function F_k(y). 
    """
    
    return 0.5 * (np.linalg.norm(x-y)**2) + (0.5/gamma_k)*(np.linalg.norm(positive_part(H(y,y_k,h)))**2) + beta*(np.linalg.norm(y-y_k)**2)

# Algorithm: Approach 2 for C = R_{-}^m with Scipy

def AlgorithmApproach2_Scipy(n, x, y_0, h, beta, gamma, eps, method = "BFGS"):
    """ 
    **Perform Approach 2 for solving the optimization problem C = R_{-}^m. using Scipy**

    Parameters:
    - n (int): Dimension of the optimization variable.
    - x (array): Input data for the optimization problem, we expect x to be projected onto h^{-1}(C).
    - y_0 (array): Initial guess for the optimization variable.
    - h (function): Function to evaluate the optimization variable h(y) \in C, must have input an n-dimensional numpy array with output an m-dimensional numpy array.
    - beta (float): Parameter for the optimization problem.
    - gamma (function): Function to calculate the gamma parameter.
    - eps (float): Tolerance for stopping the optimization.
    - method (str): Optimization method to use (default is "BFGS").

    Returns:
    - y_optimal (array): Optimized variable that minimizes the objective function.
    - y_values (list): Sequence of optimized variables throughout the iterations.
    """
    
    # Auxiliary Functions
    # Function [x]_{+} 
    def positive_part(x):
        return np.maximum(0, x)
    
    # Function H(y;y_k) = h(y_k) + Dh(y_k) @ (y-y_k)
    def H(y,y_k):
        # Define the diferential Dh(\cdot) using autograd Jacobian approximation of the given h(x) function
        Dh = ag.jacobian(h)
        return h(y_k) + Dh(y_k) @ (y-y_k)
    
    # Initial iteration
    k = 0
    convergenceCondition = True
    
    # Objective function of problem (P_0)
    def F_k(y):
        return 0.5 * (np.linalg.norm(x-y)**2) + (0.5/gamma(k))*(np.linalg.norm(positive_part(H(y,y_0)))**2) + beta*(np.linalg.norm(y-y_0)**2)
        
    # Initial guess for optimization
    x_0 = np.random.uniform(-15, 15, size=(n,))
        
    # Solve problem (P_1), with Scipy, default is "BFGS"
    P_k = minimize(F_k, x_0, method= method)
        
    # List to construct sequence (y_n)
    y_values = [y_0]
    
    # Start iterating the algorithm
    while convergenceCondition: 
        # Update the iteration counter
        k += 1
        
        # Update y_{k} = y_{k+1}
        y_k = P_k.x
        
        # Objective function of problem (P_k), solve (P_k) problem to obtain y_{k+1}
        def F_k(y):
            return 0.5 * np.linalg.norm(x-y)**2 + (0.5/gamma(k))* np.linalg.norm(positive_part(H(y,y_k)))**2 + beta * np.linalg.norm(y-y_k)**2
        
        # Initial guess for optimization
        x_0 = np.random.uniform(-15, 15, size=(n,))
        
        # Solve problem (P_k) with Scipy, default is "BFGS"
        P_k = minimize(F_k, x_0, method= method)
        
        # Obtain y_{k+1} and save it in the sequence (y_values)
        y_k_plus = P_k.x
        y_values.append(y_k_plus)
        
        # Check the convergence condition ||y_{k+1} - y_{k}|| < eps to stop the optimization
        if np.linalg.norm(y_k_plus - y_k) < eps:
            convergenceCondition = False

    # Save last iteration as optimal
    y_optimal = y_k_plus
    print('Solution Obtained: y=', y_optimal, '| Evaluation in h(y) =', h(y_k_plus))
    print('Number of iterations necessary to satisfy the condition ||y_{k+1}-y_{k}|| <', eps, 'are k =', k, 'with the optimization method:', method)
    
    # Check if solution satisfies being a feasible point in h^{-1}(C)
    if all(i <= 0.1 for i in h(y_optimal)):
        print("Solution found is: Feasible, lies into h^{-1}(C)")
    else:
        print("Solution found is: Unfeasible, doesn´t lies into h^{-1}(C)")
        
    return y_optimal, y_values

# Algorithm: Approach 2 for C = R_{-}^m with CVXPY

def AlgorithmApproach2_CVXPY(n, x, y_0, h, beta, gamma, eps):
    """ Perform Approach 2 for solving the optimization problem C = R_{-}^m using CVXPY.

    Parameters:
    - n (int): Dimension of the optimization variable.
    - x (array): Input data for the optimization problem, expected to be projected onto h^{-1}(C).
    - y_0 (array): Initial guess for the optimization variable.
    - h (function): Function to evaluate the optimization variable h(y) in C.
    - beta (float): Parameter for the optimization problem.
    - gamma (function): Function to calculate the gamma parameter.
    - eps (float): Tolerance for stopping the optimization.

    Returns:
    - y_optimal (array): Optimized variable that minimizes the objective function.
    - y_values (list): Sequence of optimized variables throughout the iterations. """
    
    # Initial iteration
    k = 0
    convergenceCondition = True
    
    # CVXPY variable for optimization
    y = cp.Variable(n)
    
    # Define the differential Dh(\cdot) using autograd Jacobian approximation of the given h(x) function
    Dh = ag.jacobian(h)
        
    # Objective function
    objective_initial = 0.5 * cp.sum_squares(x - y) + (0.5 / gamma(k)) * cp.sum_squares(cp.pos(h(y_0) + Dh(y_0) @ (y - y_0))) + beta * cp.sum_squares(y - y_0)

    # Create the CVXPY problem
    problem = cp.Problem(cp.Minimize(objective_initial))
    
    # Solve the CVXPY problem
    problem.solve()
    
    # List to construct sequence (y_n)
    y_values = [y_0]

    # Solve the optimization problem until convergence
    while convergenceCondition:
        # Update the iteration counter
        k += 1
    
        # Update y_{k} = y_{k+1} and \gamma_k in each iteration
        y_k = y.value
        
        # Update the objective function for the new iteration, cp.pos the is positive part
        objective = 0.5 * cp.sum_squares(x - y) + (0.5 / gamma(k)) * cp.sum_squares(cp.pos(h(y_k) + Dh(y_k) @ (y - y_k))) + beta * cp.sum_squares(y - y_k)

        # Update the CVXPY problem
        problem = cp.Problem(cp.Minimize(objective))

        # Solve the updated problem
        problem.solve()

        # Obtain y_{k+1} and save it in the sequence (y_values)
        y_k_plus = y.value
        y_values.append(y_k_plus)

        # Check the convergence condition ||y_{k+1} - y_{k}|| < eps to stop the optimization
        if np.linalg.norm(y_k_plus - y_k) < eps:
            convergenceCondition = False

    # Save last iteration as optimal
    y_optimal = y_k_plus
    print('Solution Obtained: y=', y_optimal, '| Evaluation in h(y) =', h(y_k_plus))
    print('Number of iterations necessary to satisfy the condition ||y_{k+1}-y_{k}|| <', eps, 'are k = ', k)

    # Check if solution satisfies being a feasible point in h^{-1}(C)
    if all(i <= 0.01 for i in h(y_optimal)):
        print("Solution found is: Feasible, lies into h^{-1}(C)")
    else:
        print("Solution found is: Unfeasible, doesn´t lies into h^{-1}(C)")
        
    return y_optimal, y_values

### Algorithm Composite Sequence

def CompositeSequence(n, x, y_0, h, gamma, eps, method = "BFGS"):
    """ 
    **Solve sequence of composite problems and return (y_gamma)**

    Parameters:
    - n (int): Dimension of the optimization variable.
    - x (array): Input data for the optimization problem, we expect x to be projected onto h^{-1}(C).
    - y_0 (array): Initial guess for the optimization variable.
    - h (function): Function to evaluate the optimization variable h(y) \in C, must have input an n-dimensional numpy array with output an m-dimensional numpy array.
    - gamma (function): Function to calculate the gamma parameter.
    - eps (float): Tolerance for stopping the optimization.
    - method (str): Optimization method to use (default is "BFGS").

    Returns:
    - y_values (list): Sequence of optimized variables throughout the iterations.
    """
    
    # Auxiliary Functions
    # Function [x]_{+} 
    def positive_part(x):
        return np.maximum(0, x)
    
    # Initial iteration
    k = 0
    convergenceCondition = True
    y_values = [y_0]
    
    # Objective function of problem (P_k), solve (P_k) problem to obtain y_{k+1}
    def F_k(y):
        return 0.5 * np.linalg.norm(x-y)**2 + (0.5/gamma(k))* np.linalg.norm(positive_part(h(y)))**2
        
    # Initial guess for optimization
    x_0 = np.random.uniform(-15, 15, size=(n,))
        
    # Solve problem (P_k) with Scipy, default is "BFGS"
    P_k = minimize(F_k, x_0, method= method)
        
    # Obtain y_{k} and save it in the sequence (y_values)
    y_k = P_k.x
    y_values.append(y_k)
    
    # Start iterating the algorithm
    while convergenceCondition: 
        # Update the iteration counter
        k += 1
        
        # Objective function of problem (P_k), solve (P_k) problem to obtain y_{k+1}
        def F_k(y):
            return 0.5 * np.linalg.norm(x-y)**2 + (0.5/gamma(k))* np.linalg.norm(positive_part(h(y)))**2
        
        # Initial guess for optimization
        x_0 = np.random.uniform(-15, 15, size=(n,))
        
        # Solve problem (P_k) with Scipy, default is "BFGS"
        P_k = minimize(F_k, x_0, method= method)
        
        # Obtain y_{k} and save it in the sequence (y_values)
        y_k = P_k.x
        y_values.append(y_k)
        
        # Check if the convergence condition ||y_{k+1} - y_{k}|| < eps to stop the optimization
        if np.linalg.norm(y_values[-1] - y_values[-2]) < eps:
            convergenceCondition = False
            print('Solution Obtained: y=', y_values[-1], '| Evaluation in h(y) =', h(y_values[-1]))
            print('Number of iterations necessary to satisfy the condition ||y_{k+1}-y_{k}|| <', eps, 'are k =', k, 'with the optimization method:', method)
            
            # Check if solution satisfies being a feasible point in h^{-1}(C)
            if all(i <= 0.01 for i in h(y_values[-1])):
                print("Solution found is: Feasible, lies into h^{-1}(C)")
            else:
                print("Solution found is: Unfeasible, doesn´t lies into h^{-1}(C)")
        
        # If sequence is not converging, then break while condition
        elif k > 1000:
            break
        
    return y_values