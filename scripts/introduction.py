import os
import tensorflow as tf

# set tensorflow log level to error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# set up a pair of output only nodes and print:

node1 = tf.constant(3.0, dtype=tf.float32)   # explicitly set node type to 32 bit float
node2 = tf.constant(4.0)                     # node type is implicitly set to 32 bit float
print("[node1, node2] = ", [node1, node2])

# to evaluate nodes, we need to run the computational graph within a tf.session:

sess = tf.Session()
print("sess.run([node1, node2]) = ", sess.run([node1, node2]))

# we can build more complicated computations by combining Tensor nodes:

node3 = tf.add(node1, node2)
print("node3 = ", node3)
print("sess.run(node3) = ", sess.run(node3))

# to accept inputs to a graph we can use placeholders to promise a value at a later stage:

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b              # shorthand for tf.add(a, b)

print("adder_node = ", adder_node)

# we can then evaluate these within the context of a session:

print("sess.run(adder_node, {a: 3, b: 4.5}) = ", sess.run(adder_node, {a: 3, b: 4.5}))
print("sess.run(adder_node, {a: [1, 5], b: [2, 4]}) =", sess.run(adder_node, {a: [1, 5], b: [2, 4]}))

# we can make graphs more complicated by adding more complex operations:

add_and_triple = adder_node * 3.
print("sess.run(add_and_triple, {a: 3, b: 4.5}) = ", sess.run(add_and_triple, {a: 3, b: 4.5}))

# to be able to make a model trainable we need to be able to provide variable values

W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)

linear_model = W * x + b

print("linear_model = W * x + b")

# constants are initialized when you call tf.constant and can not change.
# by contrast variables are not initialized until you call tf.global_variables_initializer()

init = tf.global_variables_initializer()
sess.run(init)

# once initialized we can evaluate linear_model for several values of x

print("sess.run(linear_model, {x: [1, 2, 3, 4]}) = ", sess.run(linear_model, {x: [1, 2, 3, 4]}))

# obviously without target values we have no feeling for how good our model is
# lets define a classic squared difference function

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

print("sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}) = ", sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

# knowing our model is not very good we can reassign our variables to the ideal values and rerun the loss function

fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])

print("sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}) = ", sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))