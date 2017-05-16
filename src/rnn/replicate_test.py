import tensorflow as tf # type: ignore
import numpy as np # type: ignore
import argparse
import os

try:
  import cPickle as pickle # type: ignore
except:
  import pickle # type: ignore

from rnn_model import Model,RNNMode
from data_loader import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--save-dir", type=str, default='save',
  help="model directory containing checkpointed models")
parser.add_argument("--data-dir", type=str, default='data/chorales',
  help="directory containing corpus on which to test the rnn")

args = parser.parse_args()

with open(os.path.join(args.save_dir, "config.pkl"), "rb") as f:
  saved_config = pickle.load(f)
with open(os.path.join(args.save_dir, "events_vocab.pkl"), "rb") as f:
  events, vocab = pickle.load(f)

print("Loading data...")

data_mode = saved_config.mode
data_dir = args.data_dir
batch_size = saved_config.batch_size
seq_length = saved_config.seq_length
loader = DataLoader(data_mode, data_dir, batch_size, seq_length)

print("Loading model...\n")

model = Model(saved_config, op_mode=RNNMode.SAMPLE)
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  saver = tf.train.Saver(tf.global_variables())
  ckpt = tf.train.get_checkpoint_state(args.save_dir)
  if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)

    x,y = loader.test_batch
    src_events = x.flatten()
    target_evs = y.flatten()
    clk = loader.test_clock.flatten()

    net_losses = []
    dist_losses = []
    num_events = 0

    override_bs = { model.batch_size: 1 }
    state = sess.run(model.initial_multicell_state, feed_dict=override_bs)
    for src,target,tick in zip(src_events, target_evs, clk):
      xx = np.array([[src]])
      yy = np.array([[target]])
      clk = np.array([[tick]])

      feed = {
          model.input_data: xx,
          model.target_data: yy,
          model.clock_input: clk
      }
      for i,(c,h) in enumerate(model.initial_multicell_state):
        feed[c] = state[i].c
        feed[h] = state[i].h

      state, probs, loss = sess.run([model.final_state, model.probs, model.loss] , feed_dict=feed)
      dist = probs[0]
      xent = -np.log(dist[target])
      net_losses.append(loss)
      dist_losses.append(xent)

    net_mean = np.mean(np.array(net_losses))
    dist_mean = np.mean(np.array(dist_losses))

    print("num events:", len(net_losses))
    print(" net mean:", net_mean)
    print("dist mean:",dist_mean)
    print(" net losses:", net_losses[0:6], "...", net_losses[-6:])
    print("dist losses:", dist_losses[0:6], "...", dist_losses[-6:])



