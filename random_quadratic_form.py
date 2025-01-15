#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: fourier
"""

import autograd.numpy as np  # Thinly-wrapped numpy
import autograd as ag
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint
import matplotlib.pyplot as plt

# Quadratic form function
def h(x, A, b, c):
    """
    **Quadratic form function.**

    Parameters:
        x (numpy.ndarray): Input vector.
        A (numpy.ndarray): Symmetric matrix.
        b (numpy.ndarray): Vector.
        c (float): Scalar.

    Returns:
        float: Value of the quadratic form function at point x.
    """
    return 0.5 * np.dot(x.T, np.dot(A, x)) + np.dot(b.T, x) + c

def symmetric_matrix(eigenvalues):
    """
    **Generate a random symmetric matrix with given eigenvalues.**

    Parameters:
        eigenvalues (list): List of eigenvalues.

    Returns:
        tuple: Tuple containing the generated symmetric matrix and the orthogonal matrix used for generation.
    """
    n = len(eigenvalues)
    
    D = np.diag(eigenvalues)
    random_matrix = np.random.rand(n, n) 
    Q, _ = np.linalg.qr(random_matrix)
    symm_matrix = np.dot(np.dot(Q, D), Q.T)
    
    return symm_matrix, Q

def objetive_function_aux(y, x):
    """
    **Auxiliary objective function.**

    Parameters:
        y (numpy.ndarray): Input vector.
        x (numpy.ndarray): Input vector.

    Returns:
        float: Value of the auxiliary objective function 0.5*||x-y||^{2} at point x.
    """
    return 0.5 * (np.linalg.norm(x - y) ** 2)

def generate_tuple_eigenvalues(n):
    """
    **Generate a tuple of eigenvalues with at least one negative value.**

    Parameters:
        n (int): Number of eigenvalues.

    Returns:
        tuple: Tuple of eigenvalues.
    """
    n = max(1, n)
    array = np.random.randint(-10, 11, n - 2)
    array = np.append(array, np.random.randint(-15, 0))
    array = np.append(array, np.random.randint(0, 15))
    np.random.shuffle(array)
    return tuple(array)

def generate_random_parameters(n):
    """
    **Generate random parameters for a quadratic form function.**

    Parameters:
        n (int): Dimension of the quadratic form.

    Returns:
        tuple: Tuple containing the symmetric matrix A, vector b, and scalar c.
    """
    eigenvalues = generate_tuple_eigenvalues(n)
    A, _ = symmetric_matrix(eigenvalues)
    b = np.random.rand(n)
    c = np.random.uniform(-50, 50)
    return A, b, c

def random_weakly_h_test(n):
    """
    **Generate a random weakly convex quadratic form function.**

    Parameters:
        n (int): Dimension of the quadratic form.

    Returns:
        function: Random weakly convex quadratic form function.
    """
    A, b, c = generate_random_parameters(n)
    def h_test(x):
        return h(x, A, b, c)
    return h_test

def generate_nonlinear_setting(n, m):
    """
    **Generate nonlinear constraints for optimization.**

    Parameters:
        n (int): Dimension of the quadratic form.
        m (int): Number of constraints.

    Returns:
        tuple: Tuple containing the combined constraint function and NonlinearConstraint object.
    """
    h_funcs = [random_weakly_h_test(n) for _ in range(m)]
    def h_combined(x):
        return np.array([h(x) for h in h_funcs])
    upper_bounds = np.zeros(len(h_funcs))
    lower_bounds = -np.inf * np.ones(len(h_funcs))
    constraint = NonlinearConstraint(h_combined, lower_bounds, upper_bounds)
    return h_combined, constraint

def find_x_given_multiple_restriction(h_combined, n, m):
    """
    **Find a point satisfying multiple constraints.**

    Parameters:
        h_combined (function): Combined constraint function.
        n (int): Dimension of the quadratic form.
        m (int): Number of constraints.

    Returns:
        numpy.ndarray: Point satisfying the constraints.
    """
    upper_bounds = np.inf * np.ones(m)
    lower_bounds = -np.inf * np.ones(m)
    lower_bounds[0] = 0.5
    constraint = NonlinearConstraint(h_combined, lower_bounds, upper_bounds)
    initial_guess = np.ones(n)
    def objective_function(x):
        return 0
    result = minimize(objective_function, initial_guess, method='trust-constr', constraints=constraint)
    x_given = result.x
    return x_given

def check_intersection_non_empty(constraint, n):
    """
    **Check if the intersection of constraints is non-empty.**

    Parameters:
        constraint (NonlinearConstraint): NonlinearConstraint object.
        n (int): Dimension of the quadratic form.

    Returns:
        bool: True if the intersection is non-empty, False otherwise.
    """
    def objective_function(x):
        return 0
    initial_guess = np.ones(n)
    result = minimize(objective_function, initial_guess, method='trust-constr', constraints=constraint)
    if result.success:
        return True
    else:
        return False

from tqdm import tqdm

def count_feasible_solutions_multiple_restriction(n, m, k):
    """
    **Perform optimization and count feasible solutions among multiple feasible problems.**

    Parameters:
        n (int): Dimension the state space.
        m (int): Dimension of the constraints sets.
        k (int): Number of problems to solve.

    Returns:
        tuple: A tuple containing:
            - The number of feasible problems encountered.
            - The number of infeasible problems encountered.
            - The count of feasible solutions obtained.
    """
    tolerance = 0.000000001
    feasible_problem = 0
    feasible_count = 0
    
    for _ in tqdm(range(k), desc="Progress", unit="Problem"):
        h_combined, constraint = generate_nonlinear_setting(n, m)
        
        if check_intersection_non_empty(constraint):
            feasible_problem += 1
            x_given = find_x_given_multiple_restriction(h_combined, n, m)  
            x_0 = np.random.uniform(-10, 10, n)
            def objective_function(y):
                return objetive_function_aux(y, x_given)
            sol = minimize(objective_function, x_0, method='trust-constr', constraints=constraint)
            if all(i <= tolerance for i in h_combined(sol.x)):
                feasible_count += 1
            else:
                print("Infeasible solution found!: Problem", _, "/", k,
                      "| Solution obtained: x =", sol.x,
                      "| Evaluation h(x) =", h_combined(sol.x))
                
    unfeasible_problem = k - feasible_problem
    return feasible_problem, unfeasible_problem, feasible_count