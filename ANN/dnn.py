import os
import tensorflow as tf
import numpy as np

# set tensorflow log level to error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# CONSTRUCTION

# set constants
n_inputs = 28 * 28  # set by MNIST data set
n_hidden1 = 300  # 300 neurons in first hidden layer
n_hidden2 = 100  # 100 neurons in second hidden layer
n_outputs = 10  # classify digits into categories 0-9
learning_rate = 0.01  # controls speed of convergence

# set up placeholders
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")


# define helper function to set up neuron layers
def neuron_layer(X, n_neurons, name, activation=None):
    with tf.name_scope(name):
        input_shape = int(X.get_shape()[1])  # Get dim of input
        stddev = 2 / np.sqrt(input_shape)
        init = tf.truncated_normal((input_shape, n_neurons), stddev=stddev)  # Set up norm-dist to initialize weights
        W = tf.Variable(init, name="kernel")
        b = tf.Variable(tf.zeros([n_neurons]), name="bias")  # Set up bias node
        Z = tf.matmul(X, W) + b
        if activation is not None:
            return activation(Z)
        return Z


# set up DNN
with tf.name_scope("dnn"):
    hidden1 = neuron_layer(X, n_hidden1, name="hidden1", activation=tf.nn.relu)
    hidden2 = neuron_layer(hidden1, n_hidden2, name="hidden2", activation=tf.nn.relu)
    logits = neuron_layer(hidden2, n_outputs, name="outputs")

# set up loss classifier
with tf.name_scope("loss"):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(cross_entropy, name="loss")

# define optimizer
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

# define performance evaluation
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)  # returns 1D tensor of boolean values
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))  # cast to floats and calculate average

# initialize and set saver node
init = tf.global_variables_initializer()
saver = tf.train.Saver()


# TRAINING

# download train data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data")

# define constants
n_epochs = 40
batch_size = 50
iterations = mnist.train.num_examples // batch_size

# run session

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(iterations):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: mnist.test.images, y: mnist.test.labels})
        print(epoch, "Train Accuracy:", acc_train, "Test Accuracy:", acc_test)
    save_path = saver.save(sess, "./my_model_final.ckpt")

# Predict

with tf.Session() as sess:
    saver.restore(sess, save_path)
    X_new_scaled = mnist.test.images[:20]
    Z = logits.eval(feed_dict={X: X_new_scaled})
    y_pred = np.argmax(Z, axis=1)

print("Predicted classes:", y_pred)
print("Actual classes:   ", mnist.test.labels[:20])