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
batch_size = 100
n_batches = int(np.ceil(m / batch_size))

# set up tf nodes as placeholders
X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")

# set up a variable which holds random values in range [-1, 1]
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")

# set up y predictor
y_predictor = tf.matmul(X, theta, name="predictions")

# set up standard error and mean squared error
error = y_predictor - y
mse = tf.reduce_mean(tf.square(error), name="mse")

# use an optimizer to calculate training operation
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

# initialize variables
init = tf.global_variables_initializer()


# define function to get next batch of data
def fetch_batch(epoch, batch_index, batch_size):
    np.random.seed(epoch * n_batches + batch_index)
    indices = np.random.randint(m, size=batch_size)
    X_batch = scaled_housing_data_plus_bias[indices]
    y_batch = housing.target.reshape(-1, 1)[indices]
    return X_batch, y_batch


# train model
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
    best_theta = theta.eval()
    print(best_theta)


