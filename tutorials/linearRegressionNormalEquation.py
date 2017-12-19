import tensorflow as tf
import numpy as np
import os

from sklearn.datasets import fetch_california_housing

# set tensorflow log level to error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# we can use the normal equation to provide an analytical solution to a linear regression:

# set up training data
housing = fetch_california_housing()
m, n = housing.data.shape

# add bias node
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]

# set up tf nodes to hold training and target data
X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")

# transpose X and set up normal equation (theta = (XT . X)^-1 . XT . y)
XT = tf.transpose(X)
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)

with tf.Session() as sess:
    theta_value = theta.eval()
    print(theta_value)
