import argparse
import tensorflow as tf
import numpy as np
import time
import os

try:
  import cPickle as pickle
except:
  import pickle

from tensorflow.contrib.tensorboard.plugins import projector

from rnn_model import Model, ModelConfig
from data_loader import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="music", 
  choices=["music", "text"],
  help="music: RNN for music generation, text: char-level RNN")
parser.add_argument("--data-dir", type=str, default="data/hp/hp134",
  help="data directory containing input.txt")
parser.add_argument("--save-dir", type=str, default="save",
  help="directory to store checkpointed models")
parser.add_argument("--log-dir", type=str, default="save",
  help="directory to store tensorboard summary logs")
parser.add_argument("--save-every", type=int, default=500,
  help="save every n steps")
parser.add_argument("--batch-size", type=int, default=40,
  help="number of text chunks to simultaneously feed to the model")
parser.add_argument("--seq-length", type=int, default=40,
  help="length of text chunks to feed to RNN")
parser.add_argument("--init-from", type=str, default=None,
  help=
  """continue training from saved model at this path. 
  path must contain the following files (saved by a previous training session):
   - 'config.pkl'      : configuration
   - 'events_vocab.pkl' : vocabulary obtained from input
   - 'checkpoint'      : tf list of checkpoints
   - 'model.ckpt-*'    : actual model checkpoint files
  """)


args = parser.parse_args()
mode = DataLoader.Mode.MUSIC if args.mode == "music" else DataLoader.Mode.CHAR
loader = DataLoader(mode, args.data_dir, args.batch_size, args.seq_length)
config = ModelConfig(loader)

ckpt = None

if args.init_from is not None:
  assert os.path.isdir(args.init_from),\
    "%s must be a directory" % args.init_from
  assert os.path.isfile(os.path.join(args.init_from, "config.pkl")),\
    "could not find config.pkl"
  assert os.path.isfile(os.path.join(args.init_from, "events_vocab.pkl")),\
    "could not find events_vocab.pkl"
  ckpt = tf.train.get_checkpoint_state(args.init_from)
  assert ckpt, "No checkpoint found"
  assert ckpt.model_checkpoint_path, "No model path found in checkpoint"

  # check that the saved config is compatible with the current config
  with open(os.path.join(args.init_from, "config.pkl"), "rb") as f:
    saved_model_args = pickle.load(f)
  to_check = ["mode","hidden_size","num_layers","seq_length"]
  for check in to_check:
    assert vars(saved_model_args)[check] == vars(config)[check],\
      "model config does not match loaded config"

  # check if saved vocabulary is compatible with model
  with open(os.path.join(args.init_from, "events_vocab.pkl"), 'rb') as f:
    saved_events, saved_vocab = pickle.load(f)
  assert saved_events == loader.events,\
    "Data and loaded model disagree on event symbols!"
  assert saved_vocab == loader.vocab,\
    "Data and loaded model disagree on vocab!"

# dump model config and the text loader's events/vocab
with open(os.path.join(args.save_dir, "config.pkl"), "wb") as f:
  pickle.dump(config, f)
with open(os.path.join(args.save_dir, "events_vocab.pkl"),"wb") as f:
  pickle.dump((loader.events, loader.vocab), f)


print("Initialising model, constructing graph...")
model = Model(config)
print("Starting session.")

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  summary_op = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter(args.log_dir, graph=sess.graph)
  saver = tf.train.Saver(tf.global_variables())

  # set up RNN input embedding for visualisation in Tensorboard
  proj_config = projector.ProjectorConfig()
  emb = proj_config.embeddings.add()
  emb.tensor_name = "embedding"
  emb.metadata_path = os.path.join(args.data_dir, "event_metadata.tsv")
  projector.visualize_embeddings(train_writer, proj_config)

  if args.init_from is not None:
    saver.restore(sess, ckpt.model_checkpoint_path)

  prev_epoch_mean_loss = None

  for e in range(config.num_epochs):
    model.assign_lr(sess, config.learning_rate * (config.lr_decay ** e))
    loader.reset_batch_pointer()
    state = sess.run(model.initial_multicell_state)

    total_perplexity = 0.0

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

      summaries, loss, state, _ = \
        sess.run([summary_op, model.cost, model.final_state, model.train_op], feed)
      
      end = time.time()

      total_perplexity += loss

      global_step = e * loader.num_batches + b
      total_steps = config.num_epochs * loader.num_batches

      train_writer.add_summary(summaries, global_step)

      lr = config.learning_rate * (config.lr_decay ** e)
      print("{}/{} (epoch {}), loss = {:.3f}, time/batch = {:.3f}, lr = {:.4f}" \
        .format(global_step, total_steps, e, loss, end - start, lr))

      regular_save_step = (global_step % args.save_every) == 0
      very_last_step = (e == config.num_epochs-1 and b == loader.num_batches-1)
      if regular_save_step or very_last_step:
        checkpoint_path = os.path.join(args.save_dir, "model.ckpt")
        saver.save(sess, checkpoint_path, global_step=global_step)
        print("model saved to {}".format(checkpoint_path))

    # epoch stats
    mean_loss = total_perplexity / loader.num_batches
    print("--> epoch {} end. mean loss: {}.".format(e, mean_loss))
    if prev_epoch_mean_loss is not None:
      print("--> epoch improvement:", prev_epoch_mean_loss - mean_loss)
    prev_epoch_mean_loss = mean_loss
