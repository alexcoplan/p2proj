"""
This is a slightly modified verison of:
  https://github.com/sherjilozair/char-rnn-tensorflow/blob/master/utils.py
  Copyright (c) 2015 Sherjil Ozair

However, this code is for implementing a char-level RNN, so will not be used in
the final project.
"""

import codecs
import os
import collections
import numpy as np

try:
   import cPickle as pickle
except:
   import pickle

class TextLoader():
  def __init__(self, data_dir, batch_size, seq_length, encoding='utf-8'):
    self.data_dir = data_dir
    self.batch_size = batch_size
    self.seq_length = seq_length
    self.encoding = encoding

    input_file = os.path.join(data_dir, "input.txt")
    vocab_file = os.path.join(data_dir, "vocab.pkl")
    tensor_file = os.path.join(data_dir, "data.npy")

    if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
      print("reading text file")
      self.preprocess(input_file, vocab_file, tensor_file)
    else:
      print("loading preprocessed files")
      self.load_preprocessed(vocab_file, tensor_file)

    self.create_batches()
    self.reset_batch_pointer()

  def preprocess(self, input_file, vocab_file, tensor_file):
    with codecs.open(input_file, "r", encoding=self.encoding) as f:
      data = f.read()

    # sort characters by most to least common in dataset
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    self.chars, _ = zip(*count_pairs)
    print("Chars:", self.chars)

    # vocab is an enumeration of the characters in the dataset
    self.vocab_size = len(self.chars)
    self.vocab = dict(zip(self.chars, range(len(self.chars))))
    print("Vocab:", self.vocab)
    with open(vocab_file, 'wb') as f:
      pickle.dump(self.chars, f)

    # encode the dataset using the enumeration in self.vocab
    self.tensor = np.array(list(map(self.vocab.get, data)))
    np.save(tensor_file, self.tensor)

  def load_preprocessed(self, vocab_file, tensor_file):
    with open(vocab_file, 'rb') as f:
      self.chars = pickle.load(f)

    self.vocab_size = len(self.chars)
    self.vocab = dict(zip(self.chars, range(len(self.chars))))
    print("Restored vocab:", self.vocab)
    self.tensor = np.load(tensor_file)

  def create_batches(self):
    self.num_batches = self.tensor.size // (self.batch_size * self.seq_length)
    print("Working with %s batches" % self.num_batches)
    if self.num_batches == 0:
      assert False, "Not enouch data. Decrease seq_length and/or batch_size."

    # clip data so that it divides exactly among batches
    self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
    xdata = self.tensor
    ydata = np.copy(self.tensor)

    # make ydata a delayed copy of xdata
    # since we want the RNN to predict the next item in the sequence
    ydata[:-1] = xdata[1:]
    ydata[-1] = xdata[0] # wrap around at end

    # split data into batches
    self.x_batches = np.split(xdata.reshape(self.batch_size, -1),
    self.num_batches, 1)

    self.y_batches = np.split(ydata.reshape(self.batch_size, -1),
    self.num_batches, 1)

  def next_batch(self):
    x,y = self.x_batches[self.pointer], self.y_batches[self.pointer]
    self.pointer += 1
    return x,y

  def reset_batch_pointer(self):
    self.pointer = 0


loader = TextLoader("data", 10, 200)

for batch_num in range(2):
  xs, ys = loader.next_batch()
  for idx,arr in enumerate(xs):
    batch_str = ""
    for code in arr:
      batch_str += loader.chars[code]

    print("\n=== batch[%s], seq[%s] ===\n" % (batch_num,idx))
    print(batch_str)


