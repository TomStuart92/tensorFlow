import tensorflow as tf
import numpy as np
import os

from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

# set tensorflow log level to error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# code is as in linearRegressionGradientDescent.py

housing = fetch_california_housing()
m, n = housing.data.shape
scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]
n_epochs = 1000
learning_rate = 0.01
X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name="theta")
y_predictor = tf.matmul(X, theta, name="predictions")
error = y_predictor - y
mse = tf.reduce_mean(tf.square(error), name="mse")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)
init = tf.global_variables_initializer()

# create a saver node at end of construction
saver = tf.train.Saver()

# can also specify only to save specific vars:
# saver = tf.train.Saver({"weights": theta})

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE =", mse.eval())
            saver.save(sess, "/tmp/my_model.ckpt")  # save session data
        sess.run(training_op)
    best_theta = theta.eval()
    saver.save(sess, "/tmp/my_model_final.ckpt")  # save session data
    print(best_theta)


# to restore session data:
# with tf.Session() as sess:
#     saver.restore(sess, "/tmp/my_model_final.ckpt")


# graph gets saved in a .meta file by default. To restore graph as well as weights:
# saver = tf.train.import_meta_graph("/tmp/my_model_final.ckpt.meta")
# with tf.Session() as sess:
#     saver.restore(sess, "/tmp/my_model_final.ckpt")