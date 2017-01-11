import argparse
import tensorflow as tf
import numpy as np
import time
import os

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
parser.add_argument("--save-every", type=int, default=500,
  help="save every n steps")
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


print("Initialising model, constructing graph...")
model = Model(config)
print("Starting session.")

# TODO: code to restore model

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  saver = tf.train.Saver(tf.global_variables())

  for e in range(config.num_epochs):
    model.assign_lr(sess, config.learning_rate * (config.lr_decay ** e))
    loader.reset_batch_pointer()
    state = sess.run(model.initial_multicell_state)

    # note that we preserve the state across batches in this inner loop
    for b in range(loader.num_batches):
      start = time.time()
      x,y = loader.next_batch()
      feed = { model.input_data: x, model.target_data: y }
      
      # for each cell in the initial state for our MultiRNNCell
      # we feed the correct bit of the initial state to the corresponding
      # (c,h) tensors (c is the lstm state, h is the hidden state)
      for i, (c,h) in enumerate(model.initial_multicell_state):
        feed[c] = state[i].c
        feed[h] = state[i].h

      loss, state, _ = \
        sess.run([model.cost, model.final_state, model.train_op], feed)
      
      end = time.time()

      global_step = e * loader.num_batches + b
      total_steps = config.num_epochs * loader.num_batches

      lr = config.learning_rate * (config.lr_decay ** e)
      print("{}/{} (epoch {}), loss = {:.3f}, time/batch = {:.3f}, lr = {}" \
        .format(global_step, total_steps, e, loss, end - start, lr))

      regular_save_step = (global_step % args.save_every) == 0
      very_last_step = (e == config.num_epochs-1 and b == loader.num_batches-1)
      if regular_save_step or very_last_step:
        checkpoint_path = os.path.join(args.save_dir, "model.ckpt")
        saver.save(sess, checkpoint_path, global_step=global_step)
        print("model saved to {}".format(checkpoint_path))


