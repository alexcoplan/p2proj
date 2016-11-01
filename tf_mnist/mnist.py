from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

###################################
## Set up model
###################################

# x is a *placeholder* - a 2D tensor of floating-point numbers
# we flatten each MNIST image into a 784-dimensional vector; here,
# None indicates that this dimension can be of any length (we do not
# fix a number of images)
x = tf.placeholder(tf.float32, [None, 784])

# W is our weight matrix. 784 rows (features) and 10 columns (classes)
# the 10 columns are effectively the weights for the 10 output neurons
W = tf.Variable(tf.zeros([784,10])) 

# the biases of our 10 output neurons
b = tf.Variable(tf.zeros([10]))

# we implement our single-layer model as follows:
y = tf.nn.softmax(tf.matmul(x, W) + b)

########################################
## Set up cost function and optimizer
########################################

# placeholder for the ``true'' distribution, i.e. the example y
y_ = tf.placeholder(tf.float32, [None, 10])

# a raw implementation of cross-entropy can be numerically unstable,
# so here we use the TensorFlow library function:
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
train_step = tf.train.GradientDescentOptimizer(1.5).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for i in range(1000):
  # grab 100 random examples from the data set (this effectively performs
  # stochastic gradient descent)
  batch_xs, batch_ys = mnist.train.next_batch(100)

  # train using the selection of examples. note that it is at this point that
  # the tf "placeholders" are populated
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys}) 

# Evaluate trained model

# is the hightest-probability class from our model (y) the same class as that
# from the one-hot example vector (y_)?
#
# this returns a list of booleans.
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)) 

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_:
  mnist.test.labels}))

