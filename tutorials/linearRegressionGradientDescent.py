import tensorflow as tf
import numpy as np
import os

from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

# set tensorflow log level to error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# we can use the normal equation to provide an analytical solution to a linear regression:

# set up training data
housing = fetch_california_housing()
m, n = housing.data.shape

# scale data
scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)

# add bias data
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]

# set constants
n_epochs = 1000
learning_rate = 0.01

# set up tf nodes to hold training and target data
X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")

# set up a variable which holds random values in range [-1, 1]
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name="theta")

# set up y predictor
y_predictor = tf.matmul(X, theta, name="predictions")

# set up standard error and mean squared error
error = y_predictor - y
mse = tf.reduce_mean(tf.square(error), name="mse")

# there are three ways to calculate gradients and training_ops:

# calculate gradients manually and set up training operation to move theta by learning_rate * gradient
# gradients = 2 / m * tf.matmul(tf.transpose(X), error)
# training_op = tf.assign(theta, theta - learning_rate * gradients)

# let tf calculate gradients and set up training operation to move theta by learning_rate * gradient
# gradients = tf.gradients(mse, [theta])[0]
# training_op = tf.assign(theta, theta - learning_rate * gradients)

# use an optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

# initialize variables
init = tf.global_variables_initializer()

# train model
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE =", mse.eval())
        sess.run(training_op)
    best_theta = theta.eval()
    print(best_theta)


