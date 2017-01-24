import json
import codecs
import os
from collections import Counter
from rnn_music_rep import encode_json_notes, generate_metadata_tsv
from enum import Enum
import numpy as np # type: ignore

try:
  import cPickle as pickle # type: ignore
except:
  import pickle # type: ignore

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
    train_tensor_file = os.path.join(data_dir, "train_data.npy")
    test_tensor_file = os.path.join(data_dir, "test_data.npy")
    tensor_files = (train_tensor_file, test_tensor_file)
    metadata_file = os.path.join(data_dir, "event_metadata.tsv")

    need_to_preprocess =\
      not(os.path.exists(vocab_file)
      and os.path.exists(train_tensor_file)
      and os.path.exists(test_tensor_file)
      and os.path.exists(metadata_file))

    if need_to_preprocess:
      print("loading corpus from source file")
      self.preprocess(input_file, vocab_file, tensor_files, metadata_file)
    else:
      print("loading preprocessed files...")
      self.load_preprocessed(vocab_file, tensor_files)

    self.create_test_batch()
    self.create_train_batches()
    self.reset_batch_pointer()

  def preprocess(self, input_file, vocab_file, tensor_files, metadata_file):
    corpus_events = None 

    train_tensor_file, test_tensor_file = tensor_files

    if self.mode == self.Mode.MUSIC:
      with open(input_file, "r") as f:
        corpus = json.load(f)["corpus"]
        train_corpus = corpus["train"]
        test_corpus = corpus["validate"]
        corpora = (train_corpus, test_corpus)

      corpus_events = train_events, test_events = [],[]
      for event_list,corpus in zip(corpus_events, corpora):
        for piece in corpus:
          event_list += encode_json_notes(piece["notes"])

    else: 
      with open(input_file, "r") as f:
        # for now we just take the first 10% of the input file to be a
        # validation set.
        input_str = f.read()
        ten_percent = len(input_str) // 10
        assert ten_percent > 0, "Input file too small."
        train_events = input_str[ten_percent:] # last 90%
        test_events = input_str[:ten_percent] # first 10%
        
    print("There are", len(train_events), "events in the train corpus.")
    print("There are", len(test_events), "events in the validation corpus.")

    # we compute a shared vocabulary across the train and test set. this should
    # hopefully be the same if we computed it individually anyway, but just in
    # case...
    counter = Counter(train_events) + Counter(test_events)
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
    self.train_tensor = np.array(list(map(self.vocab.get, train_events)))
    self.test_tensor = np.array(list(map(self.vocab.get, test_events)))
    np.save(train_tensor_file, self.train_tensor)
    np.save(test_tensor_file, self.test_tensor)

  def load_preprocessed(self, vocab_file, tensor_files):
    train_tensor_file, test_tensor_file = tensor_files

    with open(vocab_file, 'rb') as f:
      self.events = pickle.load(f)
    self.vocab_size = len(self.events)
    self.vocab = dict(zip(self.events, range(self.vocab_size)))
    self.train_tensor = np.load(train_tensor_file)
    self.test_tensor = np.load(test_tensor_file)

  def create_test_batch(self):
    self.num_test_examples = self.test_tensor.size // self.seq_length

    # clip data to fit into rows of length seq_length
    self.test_tensor = self.test_tensor[:self.num_test_examples * self.seq_length]
    xdata = self.test_tensor
    ydata = np.copy(self.test_tensor)

    # make ydata a delayed copy of xdata
    ydata[:-1] = xdata[1:]
    ydata[-1] = xdata[0] # wrap around

    test_xdata = xdata.reshape(-1, self.seq_length)
    test_ydata = ydata.reshape(-1, self.seq_length)
    self.test_batch = (test_xdata, test_ydata)

  def create_train_batches(self):
    self.num_batches = self.train_tensor.size // (self.batch_size * self.seq_length)

    assert self.num_batches > 0, \
      "Insufficient data. Make seq_length and/or batch_size smaller"

    # clip data so that it divides exactly among batches
    self.train_tensor = self.train_tensor[:self.num_batches * self.batch_size * self.seq_length]
    xdata = self.train_tensor
    ydata = np.copy(self.train_tensor)

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
