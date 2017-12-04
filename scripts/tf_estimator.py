import os
import tensorflow as tf
import numpy as np                  # NumPy is often used to load, manipulate and preprocess data.

# set tensorflow log level to error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# tf.estimator is a high-level TensorFlow library that simplifies the mechanics of machine learning

# declare list of features

feature_columns = [tf.feature_column.numeric_column("x", shape=[1])]        # shape = [1] sets it to a 1D vector

# an estimator is the front end to invoke training. There are many sorts. We'll use a Linear Regressor

estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)

# its best practice to use different data sets for training and evaluation. Set them up

x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])

x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])

# we then need to set up the input functions for training and evaluation.
# we have to tell the function how many batches of data (num_epochs) we want and how big each batch should be.

input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True
)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False
)

eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False
)

# we can invoke 1000 training steps by passing the input function and number of steps

estimator.train(input_fn=input_fn, steps=1000)

# once trained we can use the estimator to evaluate our training and eval sets:

train_metrics = estimator.evaluate(input_fn=train_input_fn)
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)

print("train metrics: %r"% train_metrics)
print("eval metrics: %r"% eval_metrics)