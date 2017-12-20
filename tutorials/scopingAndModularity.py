import tensorflow as tf
import os

# set tensorflow log level to error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Name Scopes allow you to group related nodes:

y_pred = tf.constant(1.0)
y = tf.constant(1.0)

with tf.name_scope("loss") as scope:
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")

print(error.op.name)
print(mse.op.name)


# Modularity and name scopes allow us to keep code dry by using functions
def relu(X):
    w_shape = (int(X.getShape()[1]), 1)
    w = tf.Variable(tf.random_normal(w_shape), name="weights")
    b = tf.Variable(0.0, name="bias")
    z = tf.add(tf.matmul(X, w), b, name="z")
    return tf.maximum(z, 0., name="relu")


n_features = 3
X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")

# create five relu's, tf will deal with ensuring proper name scoping
relus = [relu(X) for i in range(5)]
output = tf.add_n(relus, name="output")

