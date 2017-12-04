import os
import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data          # import example data

# set tensorflow log level to error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# see: https://www.tensorflow.org/get_started/mnist/beginners


def main(_):
    # read in example data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # create the model
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, W) + b

    # define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    # implement cross_entropy as our loss function
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

    # formulate training step
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    # set session and initialize variables
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # run optimizer 1000 times
    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # test trained model and print results
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


if __name__ == '__main__':
    tf.app.run(main=main, argv=[sys.argv[0]])
