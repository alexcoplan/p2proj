# multilayer perceptron in tensorflow

import argparse

parser = argparse.ArgumentParser(description="Multilayer perceptron model for MNIST")
parser.add_argument("checkpoint", nargs="?", 
help="checkpoint file from which model should be loaded")
args = parser.parse_args()

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

import sys
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

from net_util import nn_layer,scalar_summary
from mnist_util import digit_label


# hyperparameters
learning_rate = 0.001
training_epochs = 16
batch_size = 100
display_step = 1

# network params
n_hidden_1 = 256 # size of first hidden layer
n_hidden_2 = 256 # size of second hidden layer
n_input    = 784 # size of input layer (for MNIST: 28*28 = 784)
n_classes  = 10  # number of MNIST classes

# I/O to the computation graph
x = tf.placeholder(tf.float32, [None, n_input])   # input
y = tf.placeholder(tf.float32, [None, n_classes]) # target output placeholder

# create multilayer perceptron model
#
# hidden layers have ReLU activation
h1    = nn_layer(x,  n_input,    n_hidden_1, "hidden1")
h2    = nn_layer(h1, n_hidden_1, n_hidden_2, "hidden2")
model = nn_layer(h2, n_hidden_2, n_classes, "output_layer", act=tf.identity)

# useful to know this (e.g. for visualisations)
n_input_rows = tf.shape(model)[0]
  
# define loss and optimiser
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(model, y))
scalar_summary('cost', cost)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# set up tensors for visualisation
classifications_expanded = tf.expand_dims(model, -1) # add single colour channel
new_width = tf.constant(n_classes * 50)
new_img_dims = tf.pack([n_input_rows, new_width])
resized_img = tf.image.resize_images(classifications_expanded, new_img_dims,
method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
class_vis_img = tf.squeeze(resized_img)

saver = tf.train.Saver()

init_op = tf.global_variables_initializer()

def mnist_data_for_digit(n):
  imgs = mnist.test.images
  labs = mnist.test.labels

  pairs = [p for p in zip(imgs,labs) if np.array_equal(p[1], digit_label(n))]
  imgs, labs = zip(*pairs)
  return np.array(imgs), np.array(labs)

def train(sess):
  print("Training...")
  sess.run(init_op)

  all_summaries = tf.summary.merge_all()
  busy_summaries = tf.summary.merge_all(key='busylog')
  train_writer = tf.summary.FileWriter('/tmp/multilayer', sess.graph)

  for epoch in range(training_epochs):
    avg_cost = 0.0
    num_batches = mnist.train.num_examples // batch_size

    # iterate over all batches
    for i in range(num_batches):
      batch_x, batch_y = mnist.train.next_batch(batch_size)
      _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})

      avg_cost += c / num_batches # compute average cost over batches

    # display logs every display_step
    if epoch % display_step == 0:
      print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    if epoch % 5 == 0:
      summary = sess.run(all_summaries, feed_dict={x: mnist.test.images, y: mnist.test.labels})
      train_writer.add_summary(summary, epoch)

      save_path = saver.save(sess, "/tmp/model_epoch%s.ckpt" % (epoch+1))
      print("Model saved in file: %s" % save_path)
    else:
      summary = sess.run(busy_summaries, feed_dict={x: mnist.test.images, y: mnist.test.labels})
      train_writer.add_summary(summary, epoch)

  print("Training finished!")

  # test model
  correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))

  # calculate accuracy
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

def plot_classifications(sess):
  plt.figure(1)

  for i in range(9):
    imgs, labs = mnist_data_for_digit(i)
    visualisation = sess.run(class_vis_img, feed_dict={x: imgs, y: labs})
    print(sess.run(tf.shape(visualisation)))
    plt.subplot(3,3,i+1)
    plt.imshow(visualisation)

  plt.show()

def h2_activations(sess):
  plt.figure(1)

  for i in range(10):
    imgs,labs = mnist_data_for_digit(i)
    actvs = sess.run(h2, feed_dict={x: imgs, y: labs})
    plt.subplot(1,10,i+1)
    plt.imshow(actvs)

  plt.show()

# launch the graph
with tf.Session() as sess:
  if args.checkpoint:
    print("Loading model from saved file: %s" % args.checkpoint)
    saver.restore(sess, args.checkpoint)
    print("Model restored.")

    plot_classifications(sess)
  else:
    train(sess)
  
