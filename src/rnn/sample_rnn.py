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

from rnn_model import Model,RNNMode
from data_loader import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--save-dir", type=str, default='save',
  help="model directory containing checkpointed models")
parser.add_argument("-n", type=int, default=200,
  help="number of events to sample")
parser.add_argument("--prime", type=str, default=u'Harry ',
  help="text with which to prime (warm up) the RNN")
parser.add_argument("--sample", type=int, default=0,
  help="0: weighted selection, 1: greedy max, 2: weight on word boundaries")
parser.add_argument("--output-file", type=str, default="json_out/rnn.json",
  help="location of JSON file to store output")

args = parser.parse_args()

# load saved state
with open(os.path.join(args.save_dir, "config.pkl"), "rb") as f:
  saved_config = pickle.load(f)
with open(os.path.join(args.save_dir, "events_vocab.pkl"), "rb") as f:
  events, vocab = pickle.load(f)

# the second of these two imports introduces a dependency on music21
# so we lazy load this so as to not require this for a char-RNN
if saved_config.mode == DataLoader.Mode.MUSIC:
  from rnn_music_rep import decode_events,divtoken,str_to_duration
  from score_utils import render_music21

print("Loading model...\n")

model = Model(saved_config, op_mode=RNNMode.SAMPLE)
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  saver = tf.train.Saver(tf.global_variables())
  ckpt = tf.train.get_checkpoint_state(args.save_dir)
  if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)

    if saved_config.mode == DataLoader.Mode.CHAR:
      prime = list(args.prime)
      sampled = model.sample(sess, events, vocab, args.n, prime)
      print("".join(sampled))
    else:
      offset = 0
      clock = 0
      prev_dur = 0
      ts_semis = 16
      quant = saved_config.clock_quantize 
      sample = vocab["|"]
      state = None
      result : List[str] = []
      xentropies : List[float] = []
      dentropies : List[float] = []
      
      for _ in range(args.n):
        prev_str = events[sample]
        prev_dur = str_to_duration(prev_str)
        print("offset:", offset, "clock:", clock, "prev:", prev_str)
        state, xent, dent, sample = model.clocked_sample_iter(sess, sample,
            clock, state)
        xentropies.append(xent)
        dentropies.append(dent)

        event_str = events[sample]
        if event_str == "|":
          clock = 1
          if len(result) == 0:
            continue
          
          json_object = {
            "notes" : decode_events(result),
            "entropies" : {
              "cross_entropies" : xentropies,
              "dist_entropies" : dentropies
            }
          }

          with open(args.output_file, "w") as f:
            f.write(json.dumps(json_object))

          render_music21(decode_events(result))

          break # take first output

          result = []
          offset = 0
        else:
          result.append(event_str)
          if prev_dur is not None:
            offset += prev_dur
          clock = ((offset % ts_semis) // quant) + 1
        


