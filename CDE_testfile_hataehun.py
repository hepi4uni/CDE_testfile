# It implements a numerical method for solving the contour dynamics equation for alpha-patches
# The code is written in Python 3 and uses numpy and matplotlib libraries

import numpy as np
import matplotlib.pyplot as plt

# Define the parameters of the problem
N = 2 ** 12  # number of grid points
PI = np.pi # constant value pi
Omega0 = 1 # vorticity value
h = 1 / N  # grid spacing
dt = 1e-0  # time step
T = 2 * 1e-0  # final time
t = 0  # initial time

# Define a boundary curve of an initial patch with Lagrangian parameterization
theta1 = 0.01 # angle between x-axis and arm's left-end point
theta2 = theta1 / 2 # angle between x-axis and arm's right-end point[arctan(sin(x)/(1+cos(x))) = x / 2, x in (-pi,pi)]

# Define the grid angles of initial curve
S1 = list(np.arange(theta1, 2 * PI - theta1, h * (2 ** 8))) # rest part of the curve
S2 = list(np.arange(theta2, theta1, h)) # upper arm's left-end point to right-end point
S3 = list(np.arange(2 * PI - theta1, 2 * PI - theta2, h)) # lower arm's left-end point to right-end point
S = [0] + S2 + S1 + S3 + [0]# grid angles of curve

# Define the initial curve z0
def z0(alpha):
    if alpha in S1:
        return [np.cos(alpha), np.sin(alpha)] # rest part of the curve
    elif alpha in S2:
        return [2 * np.cos(theta2) - S2.index(alpha) / len(S2), np.sin(theta1)] # upper arm's left-end point to right-end point
    elif alpha in S3:
        return [np.cos(theta1) + S3.index(alpha) / len(S3), -1 * np.sin(theta1)] # lower arm's left-end point to right-end point
    else:
        return [2, 0] # right end-point of the curve

# Define the curve z by Euler method
def z(alpha, t):
    memo = {} # create an empty dictionary for memoization
    def helper(alpha, t): # define a helper function
        if (alpha, t) in memo: # base case: the result is already computed and stored
            return memo[(alpha, t)] # return the stored value
        if t == 0: # base case: t is zero
            return z0(alpha) # set the result to z0(alpha)
        else: # recursive case: t is not zero
            total = 0
            for i in range(int(t//dt)): # update the result
                total += z(alpha, i * dt) + rhs(z, alpha, i * dt) * dt
            total /= int(t//dt)
            memo[(alpha, t)] = total
            return total # return the final result
    return helper(alpha, t) # call the helper function with the given parameters

# Define the inner function in right-hand side
def inner(z, alpha, i, t): # inner = ln|z(alpha,t)-z(alpha',t)|z_alpha(alpha',t)
    memo = {} # create an empty dictionary for memoization
    def helper(z, alpha, i, t):
        if (z, alpha, i, t) in memo:
            return memo[(z, alpha, i, t)]
        else:
            z_alpha = (np.array(z(S[i+1], t)) - np.array(z(S[i], t))) / (S[i+1] - S[i]) # differentiate z by alpha
            # To except the case when alpha = alpha'
            if alpha != S[i]:
                result = np.log(np.linalg.norm(np.array(z(alpha, t)) - np.array(z(S[i], t)))) * z_alpha
                memo[(z, alpha, i, t)] = result
                return result
            else:
                result = np.array([0, 0])
                memo[(z, alpha, i, t)] = result
                return result
    return helper(z, alpha, i, t)

# Define the right-hand side function of CDE(Contour Dynamics equation)
def rhs(z, alpha, t): # rhs = -omega0/2pi * int_0^2pi inner(z,alpha,i,t)
    memo = {} # create an empty dictionary for memoization
    def helper(z, alpha, t):
        if (z, alpha, t) in memo:
            return memo[(z, alpha, t)]
        else:
            s = [0, 0] # the variable of summation
            # Use trapezoidal rule to compute the integration
            s += inner(z, alpha, 0, t) / 2 * (S[1] - S[0]) # add half of first term times h
            for i in range(1, len(S) - 1): # iterating second term to len(S)-1 term times h
                s += inner(z, alpha, i, t) * (S[i+1] - S[i])
            s += inner(z, alpha, len(S) - 2, t) / 2 * (S[-1] - S[-2]) # add half of last term times h
            result = -1 * Omega0 * s / (2 * PI)
            memo[(z, alpha, t)] = result
            return result
    return helper(z, alpha, t)

# Define the position of the curve at time T
Curve = []
for i in range(len(S)):
    Curve.append(list(z(S[i], T)))

# The points of curve in each axis
X_axis = list(Curve[i][0] for i in range(len(Curve)))
Y_axis = list(Curve[i][1] for i in range(len(Curve)))

# Plot the curve
plt.plot(X_axis, Y_axis)
plt.axis('equal')
plt.show()
