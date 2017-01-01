from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def digit_label(n):
  return np.array([0.0]*n + [1.0] + [0.0]*(9-n))

d_lab = digit_label(9)

zeros = [img for (img,lab) in zip(mnist.test.images, mnist.test.labels) if
np.array_equal(lab,d_lab)]

print("there are", len(zeros), "zeros")

z_dat = zeros[0]
img = np.array([[z_dat[x + y*28] for x in range(28)] for y in range(28)])
print(len(img))
print(len(img[0]))
plt.imshow(img)
plt.show()
