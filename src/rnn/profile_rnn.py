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
  help="path to JSON corpus from which to extract test sequence")
parser.add_argument("--output-file", type=str, default="out/rnn_evald.json")
parser.add_argument("--path", default=False, action='store_true',
  help="instead of predicting test example, run a pathalogical exmaple")
parser.add_argument("--target-idx", type=int, default=0,
  help="index of sequence in test set to profile RNN on")

args = parser.parse_args()

# load saved state
with open(os.path.join(args.save_dir, "config.pkl"), "rb") as f:
  saved_config = pickle.load(f)
with open(os.path.join(args.save_dir, "events_vocab.pkl"), "rb") as f:
  events, vocab = pickle.load(f)

# load target piece to predict
quant = 1

if args.path:
  path_len = 30
  pitches = [67] * path_len
  durs = [4] * path_len
  offsets = list(map(lambda x: x*4, range(path_len)))
  json_notes = [list(a) for a in zip(pitches,offsets,durs)]
  ts_semis = 16
  target = encode_json_notes(json_notes, quant, ts_semis)
else:
  with open(args.corpus_path, "r") as f:
    corpus = json.load(f)["corpus"]
    test_corpus = corpus["validate"]
    piece = test_corpus[args.target_idx]
    target = encode_json_notes(piece["notes"], quant, piece["time_sig_amt"])

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
  processed : List[str] = []
  xentropies : List[float] = []
  dentropies : List[float] = []
  prev_event, prev_tick = target[0][0], target[1][0]

  for event_t,tick_t in list(zip(*target))[1:]:
    processed.append(prev_event)
    encoded = vocab[prev_event]
    state, dist = model.clocked_dist_iter(sess, encoded, prev_tick, state)
    nxt_encoded = vocab[event_t]
    xent = -np.log2(dist[nxt_encoded])
    dent = -np.sum(dist * np.log2(dist))
    xentropies.append(float(xent))
    dentropies.append(float(dent))
    prev_event, prev_tick = event_t, tick_t

  json_object = {
    "notes" : decode_events(processed),
    "entropies" : {
      "cross_entropies" : xentropies,
      "dist_entropies" : dentropies
    }
  }

  print("Mean xent: {}".format(np.mean(np.array(xentropies))))
  print("Done.")

  with open(args.output_file, "w") as f:
    f.write(json.dumps(json_object))


