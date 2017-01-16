import json
import codecs
import os
from collections import Counter
from rnn_music_rep import encode_json_notes, generate_metadata_tsv
from enum import Enum
import numpy as np

try:
  import cPickle as pickle
except:
  import pickle

class DataLoader(object):
  class Mode(Enum):
    CHAR = 0
    MUSIC = 1

  def __init__(self, mode, data_dir, batch_size, seq_length):
    self.mode = mode
    self.data_dir = data_dir
    self.batch_size = batch_size
    self.seq_length = seq_length
    
    fname = "input.txt" if self.mode == self.Mode.CHAR else "corpus.json"
    input_file = os.path.join(data_dir, fname)
    vocab_file = os.path.join(data_dir, "vocab.pkl")
    tensor_file = os.path.join(data_dir, "data.npy")
    metadata_file = os.path.join(data_dir, "event_metadata.tsv")

    need_to_preprocess =\
      not(os.path.exists(vocab_file)
      and os.path.exists(tensor_file)
      and os.path.exists(metadata_file))

    if need_to_preprocess:
      print("loading corpus from source file")
      self.preprocess(input_file, vocab_file, tensor_file, metadata_file)
    else:
      print("loading preprocessed files...")
      self.load_preprocessed(vocab_file, tensor_file)

    self.create_batches()
    self.reset_batch_pointer()

  def preprocess(self, input_file, vocab_file, tensor_file, metadata_file):
    corpus_events = None 

    if self.mode == self.Mode.MUSIC:
      with open(input_file, "r") as f:
        corpus = json.load(f)["corpus"]
      corpus_events = []
      for piece in corpus:
        corpus_events += encode_json_notes(piece["notes"])
    else: 
      with open(input_file, "r") as f:
        corpus_events = f.read()

    print("There are", len(corpus_events), "total events in the corpus.")

    counter = Counter(corpus_events)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    self.events, _ = zip(*count_pairs)

    print("There are {} distinct events in the corpus.".format(len(self.events)))
    with open(vocab_file, "wb") as f:
      pickle.dump(self.events, f)

    # generate TensorBoard metadata
    with open(metadata_file, "w") as f:
      if self.mode == self.Mode.MUSIC:
        f.write(generate_metadata_tsv(self.events))
      else:
        f.write("\n".join(self.events))

    self.vocab_size = len(self.events)
    self.vocab = dict(zip(self.events, range(self.vocab_size)))
    self.tensor = np.array(list(map(self.vocab.get, corpus_events)))
    np.save(tensor_file, self.tensor)

  def load_preprocessed(self, vocab_file, tensor_file):
    with open(vocab_file, 'rb') as f:
      self.events = pickle.load(f)
    self.vocab_size = len(self.events)
    self.vocab = dict(zip(self.events, range(self.vocab_size)))
    self.tensor = np.load(tensor_file)

  def create_batches(self):
    self.num_batches = self.tensor.size // (self.batch_size * self.seq_length)

    if self.num_batches == 0:
      assert False, \
        "Insufficient data. Make seq_length and/or batch_size smaller"

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

  def inspect_batches(self, n_batches_to_inspect):
    for batch_num in range(n_batches_to_inspect):
      xs, ys = self.next_batch()
      for idx,arr in enumerate(xs):
        batch_str = ""
        for code in arr:
          batch_str += self.events[code]

        print("\n=== batch[%s], seq[%s] ===\n" % (batch_num,idx))
        print(batch_str)
