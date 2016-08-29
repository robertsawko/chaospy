'''
Analysis of convergence for a second order polynomial.
'''
from chaospy import (
    Uniform, J, orth_ttr,
    generate_quadrature, fit_quadrature, fit_regression)
from numpy import zeros
from scipy.integrate import nquad


# The model solver
def u(x, y):
    return x**2 + x * y + y**2 + x + y + 1.0

# Defining the input random distributions:
X = Uniform()
Y = Uniform()
dist = J(X, Y)


def L2norm_in_random_space2D(u, u_hat):
    return nquad(
        lambda x, y: (u(x, y) - u_hat(x, y))**2,
        [[0.0, 1.0], [0.0, 1.0]])[0]**(0.5)


def cp_pseudospectral(max_order=10, sparse=False, rule='G'):
    '''Test of the pseudospectral method'''
    errors = zeros(max_order)

    for order in range(max_order):
        P, norms = orth_ttr(order, dist, retall=True)
        nodes, weights = generate_quadrature(
            order, dist, sparse=sparse, rule=rule)
        solves = [u(s[0], s[1]) for s in nodes.T]
        u_hat = fit_quadrature(P, nodes, weights, solves, norms=norms)
        errors[order] = L2norm_in_random_space2D(u, u_hat)
    return errors


def cp_point_collocation(max_order, sampling_rule='M'):
    '''Test of the point collocation method'''
    errors = zeros(max_order)

    for order in range(max_order):
        P = orth_ttr(order, dist)
        nodes = dist.sample(len(P), rule=sampling_rule)
        solves = [u(s[0], s[1]) for s in nodes.T]
        u_hat = fit_regression(P, nodes, solves, rule='T')
        errors[order] = L2norm_in_random_space2D(u, u_hat)

    return errors

ps_t = cp_pseudospectral(8)
ps_s = cp_pseudospectral(8, sparse=True)

sampling_rules = [
    ('R', 'Random'), ('L', 'Latin-hypercube'),
    ('S', 'Sobol'), ('H', 'Halton'), ('M', 'Hammersley')]
pc_results = [
    cp_point_collocation(8, sampling_rule=sr[0]) for sr in sampling_rules]
