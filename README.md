# GradientDescent_Optimizing_LinearRegression

This code uses gradient descent (popular first-order optimization function.) as a means of performing linear regression on a dataset
of test scores vs hours studied.  Attempts to model this data linearly using the popular y = mx + b, slope-intercept formula.

### We first import our libraries and data.  Define params.
```
    import numpy as np
    points = genfromtxt("studyVSscore.csv", delimiter=",")
    scores = np.array(points[:,0])
    hours = np.array(points[:,1])

    learning_rate = 0.0001 # Optimizable learning rate
    initial_b = 0 # Initial y-intercept guess
    initial_m = 0 # Initial slope guess
    num_iterations = 1000 # Desired training iterations
```
Gradient Descent is an iterative first order optimazation, we will use 3 helper functions to perform the optimization.

### Firstly, we need a function for computing errors of predicted vs actual results. (Least squares loss function)
```
def compute_error_for_line_given_points(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + b)) ** 2  # y = mx + b, squared to make all numbers positive (least squares)
    return totalError / float(len(points))
```

### We use this loss value to iteartively optimize our gradient values with step_gradient() function
```
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
```

### Finally a function for neatly running our gradient descent. (uses step_gradient() within)
```
# Runs the step_gradient function
def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations, scores, hours):
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        b, m = step_gradient(b, m, array(points), learning_rate)
    return [b, m]
```



