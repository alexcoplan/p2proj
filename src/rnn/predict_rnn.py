import tensorflow as tf # type: ignore
import numpy as np # type: ignore
import argparse
import os
import json

from typing import List

try:
  import cPickle as pickle # type: ignore
except:
  import pickle # type: ignore

from rnn_music_rep import *
from rnn_model import Model,RNNMode
from data_loader import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--save-dir", type=str, default='save',
  help="model directory containing checkpointed models")
parser.add_argument("--corpus-path", type=str, default=None,
  help="path to JSON corpus on which to test RNN")
parser.add_argument("--output-file", type=str, default="out/rnn_evald.json")

args = parser.parse_args()

# load saved state
with open(os.path.join(args.save_dir, "config.pkl"), "rb") as f:
  saved_config = pickle.load(f)
with open(os.path.join(args.save_dir, "events_vocab.pkl"), "rb") as f:
  events, vocab = pickle.load(f)

# load target piece to predict
quant = 1

with open(args.corpus_path, "r") as f:
  corpus = json.load(f)["corpus"]
  test_corpus = corpus["validate"]
  targets = map(lambda p: encode_json_notes(p["notes"], quant,
    p["time_sig_amt"]), test_corpus)

# load model
print("Loading model...")

model = Model(saved_config, op_mode=RNNMode.SAMPLE)
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  saver = tf.train.Saver(tf.global_variables())
  ckpt = tf.train.get_checkpoint_state(args.save_dir)
  if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
  else:
    exit()

  print("Starting prediction...")

  state = None
  seq_xents = []
  for target in targets:
    xentropies : List[float] = []
    prev_event, prev_tick = target[0][0], target[1][0]

    for event_t,tick_t in list(zip(*target))[1:]:
      encoded = vocab[prev_event]
      state, dist = model.clocked_dist_iter(sess, encoded, prev_tick, state)
      nxt_encoded = vocab[event_t]
      xent = float(-np.log2(dist[nxt_encoded]))
      xentropies.append(xent)
      prev_event, prev_tick = event_t, tick_t

    seq_xents.append(float(np.mean(np.array(xentropies))))

  print("Done.")
  print("Mean xent:", np.mean(np.array(seq_xents)))

  json_object = {
    "xents" : seq_xents,
  }
  with open(args.output_file, "w") as f:
    f.write(json.dumps(json_object))


