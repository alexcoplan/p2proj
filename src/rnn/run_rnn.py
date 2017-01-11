import argparse
import tensorflow as tf
import numpy as np

try:
  import cPickle as pickle
except:
  import pickle

from rnn_model import Model, ModelConfig
from text_loader import TextLoader

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", type=str, default="data/hp",
  help="data directory containing input.txt")
parser.add_argument("--save-dir", type=str, default="save",
  help="directory to store checkpointed models")
parser.add_argument("--batch-size", type=int, default=50,
  help="number of text chunks to simultaneously feed to the model")
parser.add_argument("--seq-length", type=int, default=100,
  help="length of text chunks to feed to RNN")

args = parser.parse_args()

loader = TextLoader(args.data_dir, args.batch_size, args.seq_length)
config = ModelConfig(loader)

# dump model config and the text loader's chars/vocab
with open(os.path.join(args.save_dir, "config.pkl"), "wb") as f:
  pickle.dump(config, f)
with open(os.path.join(args.save_dir, "chars_vocab.pkl"),"wb") as f:
  pickle.dump((loader.chars, loader.vocab), f)

model = Model(config)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  saver = tf.train.Saver(tf.global_variables())

  np_initial_state = sess.run(model.initial_multicell_state)

  # TODO: code to restore model
  # TODO: training code

  
