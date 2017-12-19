import tensorflow as tf
import os

# set tensorflow log level to error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Any node you create is automatically added to the default graph
x1 = tf.Variable(1)
print(x1.graph is tf.get_default_graph())   # true

# We can create multiple independent graphs and add nodes manually:
graph = tf.Graph()

with graph.as_default():
    x2 = tf.Variable(2)

print(x2.graph is graph)    # true
print(x2.graph is tf.get_default_graph())   # false

# and can reset the default graph:
tf.reset_default_graph()

# when you evaluate a node, tf determines nodes that it depends on and evaluates those first:

w = tf.constant(3)
x = w + 2
y = x + 5
z = x * 3

with tf.Session() as sess:
    print(y.eval())     # 10
    print(z.eval())     # 15

# Important to note that tf does not reuse values, w and x are calculated twice above.
# All values are dropped between runs. Must calculate in same run to save computational time:

with tf.Session() as sess2:
    y_val, z_val = sess2.run([y, z])
    print(y_val)     # 10
    print(z_val)     # 15
