# This code uses gradient descent (popular first-order optimazation function.) as a means of performing linear regression on a dataset
# of test scores vs hours studied.  Attempts to model this data linearly using the popular y = mx + b, slope-intercept formula.

from numpy import *
import matplotlib.pyplot as plt
# y = mx + b
# m is slope, b is y-intercept

# Compute error function
def compute_error_for_line_given_points(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + b)) ** 2  # y = mx + b, squared to make all numbers positive (least squares)
    return totalError / float(len(points))

# Iterative gradient optimization
def step_gradient(b_current, m_current, points, learningRate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]

# Runs the step_gradient function
def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        b, m = step_gradient(b, m, array(points), learning_rate)
    return [b, m]

def run():
    import numpy as np
    points = genfromtxt("studyVSscore.csv", delimiter=",")
    scores = np.array(points[:,0])
    hours = np.array(points[:,1])

    learning_rate = 0.0001 # Optimizable learning rate
    initial_b = 0 # Initial y-intercept guess
    initial_m = 0 # Initial slope guess
    num_iterations = 1000 # Desired training iterations

    # Display initial slope-intercept estimate
    print ("Starting gradient descent at b = {0}, m = {1}, error = {2}"
           .format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points)))
    print ("Running...")
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)

    # Display final slope-intercept estimate
    print ("After {0} iterations b = {1}, m = {2}, error = {3}"
           .format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points)))

    print('slope formula: ', 'y = {0:.4f}(x)'.format(m) + ' {0:.4f}'.format(b))

    # Plot our data against our linear regression estimate to check fit.
    plt.scatter(scores, hours) # Plot our data with scatter points
    plt.plot(scores, scores * m + b, 'r') # Plot our linear regression estimate on top of data
    plt.title('Regression of Test Scores vs Hours Studied')
    plt.xlabel('Scores')
    plt.ylabel('Hours Studied')
    plt.show()

if __name__ == '__main__':
    run()
