import tensorflow as tf
import os

# set tensorflow log level to error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# create a simple graph
x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")
f = x * x * y + y + 2

# This does not do any calculation. Need to initialize a session:
sess = tf.Session()
sess.run(x.initializer)
sess.run(y.initializer)

result = sess.run(f)
print(result)

sess.close()

# Can initialize once as follows:
with tf.Session() as sess2:
    x.initializer.run()
    y.initializer.run()
    result = f.eval()
    print(result)

# Instead of initializing each variable individually can do globally:
init = tf.global_variables_initializer()

with tf.Session() as sess3:
    init.run()
    result = f.eval()
    print(result)

# Can also use an interactive session which doesn't need to be declared in a block, but does need to be closed:
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()

init.run()
result = f.eval()
print(result)

sess.close()
