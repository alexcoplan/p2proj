import tensorflow as tf
import numpy as np
import argparse
import os

try:
  import cPickle as pickle
except:
  import pickle

from rnn_model import Model
from rnn_model import SamplingMethod

parser = argparse.ArgumentParser()
parser.add_argument("--save-dir", type=str, default='save',
  help="model directory to store checkpointed models")
parser.add_argument("-n", type=int, default=500,
  help="number of characters to sample")
parser.add_argument("--prime", type=str, default=u'Harry ',
  help="text with which to prime (warm up) the RNN")
parser.add_argument("--sample", type=int, default=0,
  help="0: weighted selection, 1: greedy max, 2: weight on word boundaries")

args = parser.parse_args()

sampling_type = SamplingMethod(args.sample)

# load saved state
with open(os.path.join(args.save_dir, "config.pkl"), "rb") as f:
  saved_config = pickle.load(f)
with open(os.path.join(args.save_dir, "chars_vocab.pkl"), "rb") as f:
  chars, vocab = pickle.load(f)

print("Loading model...\n")

model = Model(saved_config, is_training=False)
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  saver = tf.train.Saver(tf.global_variables())
  ckpt = tf.train.get_checkpoint_state(args.save_dir)
  if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    print(model.sample(sess, chars, vocab, args.n, args.prime, sampling_type))
