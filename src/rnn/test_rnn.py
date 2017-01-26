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

model = Model(saved_config, op_mode=RNNMode.TEST)
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  saver = tf.train.Saver(tf.global_variables())
  ckpt = tf.train.get_checkpoint_state(args.save_dir)
  if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    x,y = loader.test_batch
    test_feed = { model.input_data: x, model.target_data: y }
    print("test loss:", sess.run(model.loss, feed_dict=test_feed))
