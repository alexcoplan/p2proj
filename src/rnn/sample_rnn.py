import tensorflow as tf # type: ignore
import numpy as np # type: ignore
import argparse
import os

try:
  import cPickle as pickle # type: ignore
except:
  import pickle # type: ignore

from rnn_model import Model, SamplingMethod
from data_loader import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--save-dir", type=str, default='save',
  help="model directory to store checkpointed models")
parser.add_argument("-n", type=int, default=500,
  help="number of events to sample")
parser.add_argument("--prime", type=str, default=u'Harry ',
  help="text with which to prime (warm up) the RNN")
parser.add_argument("--sample", type=int, default=0,
  help="0: weighted selection, 1: greedy max, 2: weight on word boundaries")

args = parser.parse_args()

sampling_type = SamplingMethod(args.sample)

# load saved state
with open(os.path.join(args.save_dir, "config.pkl"), "rb") as f:
  saved_config = pickle.load(f)
with open(os.path.join(args.save_dir, "events_vocab.pkl"), "rb") as f:
  events, vocab = pickle.load(f)

# the second of these two imports introduces a dependency on music21
# so we lazy load this so as to not require this for a char-RNN
if saved_config.mode == DataLoader.Mode.MUSIC:
  from rnn_music_rep import decode_events,divtoken
  from score_utils import render_music21 

print("Loading model...\n")

model = Model(saved_config, is_training=False)
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  saver = tf.train.Saver(tf.global_variables())
  ckpt = tf.train.get_checkpoint_state(args.save_dir)
  if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)

    if saved_config.mode == DataLoader.Mode.CHAR:
      prime = list(args.prime)
    else:
      prime = [divtoken]
    events = model.sample(sess, events, vocab, args.n, prime, sampling_type)
    if saved_config.mode == DataLoader.Mode.CHAR:
      print("".join(events))
    else:
      print(events)
      buff = [] # type: List[List[int]]
      for e in events[1:]:
        if e == divtoken:
          render_music21(decode_events(buff))
          buff = []
        else:
          buff.append(e)

