import tensorflow as tf # type: ignore
import numpy as np # type: ignore

from enum import Enum
from typing import List
from data_loader import DataLoader

# this is passed to Model's constructor to determine how to build the RNN
#  - TRAIN adds dropout and configures the RNN to accept batched sequences
#   in bulk. 
#  - SAMPLE configures the RNN to process one event at a time (no dropout).
#  - TEST is like TRAIN but without dropout.
class RNNMode(Enum):
  TRAIN = 0
  SAMPLE = 1
  TEST = 2

class ModelConfig(object):
  def __init__(self, data_loader : DataLoader) -> None:
    self.keep_prob = 0.5
    self.max_grad_norm = 5
    self.learning_rate = 1.0
    self.num_epochs = 60
    self.hidden_size = 320
    self.lr_decay = 0.95
    self.num_layers = 1
    self.mode         = data_loader.mode
    self.batch_size   = data_loader.batch_size
    self.seq_length   = data_loader.seq_length
    self.vocab_size   = data_loader.vocab_size

    if data_loader.mode == DataLoader.Mode.MUSIC:
      self.clock_width = data_loader.clock_width
      self.clock_quantize = data_loader.clock_quantize

def generate_clock_embedding(clock_width : int) -> List[List[int]]:
  one_hot = []
  for i in range(clock_width):
    one_hot.append([0] * i + [1] + [0] * (clock_width-i-1))
  zero_state = [0] * clock_width
  return [zero_state] + one_hot
    

class Model(object):
  rnn_dtype = tf.float32

  def __init__(self, config : ModelConfig, op_mode=RNNMode.TRAIN) -> None:
    # training hyperparams
    vocab_size = config.vocab_size
    seq_length = config.seq_length
    hidden_units = config.hidden_size # width of LSTM hidden layer

    if config.mode == DataLoader.Mode.MUSIC:
      state_size = hidden_units - config.clock_width
    else:
      state_size = hidden_units

    if op_mode == RNNMode.SAMPLE:
      seq_length = 1

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_units, state_is_tuple=True)
    if op_mode == RNNMode.TRAIN and \
       config.keep_prob < 1.0 and config.num_layers > 1:
      # apply dropout to output of each layer
      lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
          lstm_cell, output_keep_prob=config.keep_prob)

    cells = [lstm_cell] * config.num_layers
    self.cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)

    # we use None here because we want to allow different batch sizes
    # for train and test datasets
    self.input_data = tf.placeholder(tf.int32, [None, seq_length], "inputs")
    self.target_data = tf.placeholder(tf.int32, [None, seq_length], "labels")

    # calculate batch_size implicitly from input_data
    # n.b. we make this an accessible property since we sometimes need to
    # override it (e.g. to get the zero cell state without feeding data in)
    self.batch_size = tf.shape(self.input_data)[0]
    num_labels = tf.shape(self.target_data)[0]

    check_dims = tf.Assert(tf.equal(self.batch_size, num_labels),
        [self.batch_size, num_labels], name="check_batch_sizes")

    # get initial state for our LSTM cell
    self.initial_multicell_state = \
      self.cell.zero_state(self.batch_size, self.rnn_dtype)

    # might need to add with tf.device("/cpu:0") here if training on GPU
    # create embedding variable which is randomly initialised
    # the network will learn a dense representation for the input vocabulary
    embedding = tf.get_variable("embedding", 
    [vocab_size, state_size], dtype=self.rnn_dtype)
    inputs = tf.nn.embedding_lookup(embedding, self.input_data)

    # if we're in music mode, add some input neurons to the RNN to support
    # a time signature clock input
    if config.mode == DataLoader.Mode.MUSIC:
      clock_w = config.clock_width
      clock_states = clock_w + 1
      emb_dat = generate_clock_embedding(clock_w)
      clock_emb = tf.constant(emb_dat, shape=[clock_states, clock_w], 
        name="clk_embedding", dtype=tf.float32)

      self.clock_input = \
        tf.placeholder(tf.int32, [None, seq_length], "clock_input")
      clock_data = tf.nn.embedding_lookup(clock_emb, self.clock_input)
      inputs = tf.concat(2, [inputs, clock_data], name='concat_clock')
      
    # construct the graph for an unrolled RNN
    # 
    # tf.unstack here transforms the inputs into a list of tensors of length
    # seq_length
    #
    # at the ith position in the list is a tensor which contains the embeddings
    # found at sequence position i across all batches
    #
    # see the docs for tf.nn.rnn and the argument `inputs` to see why
    inputs = tf.unstack(inputs, num=seq_length, axis=1)
    outputs, state = tf.nn.rnn(self.cell, inputs,
      initial_state=self.initial_multicell_state)

    # transform the outputs back into the input tensor form
    output = tf.reshape(tf.concat(1, outputs), [-1, hidden_units])
    softmax_w = tf.get_variable("softmax_w", [hidden_units, vocab_size])
    softmax_b = tf.get_variable("softmax_b", [vocab_size])
    logits = tf.matmul(output, softmax_w) + softmax_b
    
    # enusre no. of inputs matches no. of labels before computing loss
    with tf.control_dependencies([check_dims]):
      # compute a weighted cross-entropy loss
      # note that this function applies softmax to the logits for us
      # since we want to weight all logits equally, we pass all 1s for the weights
      loss_vector = tf.nn.seq2seq.sequence_loss_by_example(
          [logits],
          [tf.reshape(self.target_data, [-1])],
          [tf.ones([self.batch_size * seq_length], dtype=self.rnn_dtype)],
          average_across_timesteps=True
      )

    # nb we are doing truncated backpropagation with the truncation point
    # set to seq_length

    # create tensor which gives our probability distribution (used for sampling)
    self.probs = tf.nn.softmax(logits)

    batch_size_f = tf.cast(self.batch_size, tf.float32)
    self.loss = tf.reduce_sum(loss_vector) / (batch_size_f * seq_length)

    tf.summary.scalar('loss', self.loss)
    self.final_state = state

    if op_mode != RNNMode.TRAIN:
      return

    self.lr = tf.Variable(0.0, trainable=False, name="learning_rate")
    tf.summary.scalar('learning_rate', self.lr)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars),
        config.max_grad_norm)
    
    optimizer = tf.train.GradientDescentOptimizer(self.lr)
    self.train_op = optimizer.apply_gradients(zip(grads, tvars)) # type: ignore
    
    self.new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
    self.lr_update = tf.assign(self.lr, self.new_lr)

  def assign_lr(self, sess, lr_value):
    sess.run(self.lr_update, feed_dict={self.new_lr: lr_value})

  # returns (state, sample) so we can use this in an iterative fashion
  # with some external code generating the correct clock values
  def clocked_sample_iter(self, sess, value : int, clock : int, state=None):
    if state == None:
      state = sess.run(self.cell.zero_state(1, self.rnn_dtype))

    x = np.array([[value]])
    clk = np.array([[clock]])
    
    feed = {
      self.input_data: x, self.initial_multicell_state: state,
      self.clock_input: clk
    }
    [probs,state] = sess.run([self.probs, self.final_state], feed)
    dist = probs[0] # reduce 2D batch tensor to 1D distribution

    cdf = np.cumsum(dist)
    total = np.sum(dist)
    sample = int(np.searchsorted(cdf, np.random.rand(1)*total))
    
    return state, sample


  def sample(self, sess, events, vocab, num, prime_events):
    # create initial state for our RNN (multi-)cell with a batch size of one
    state = sess.run(self.cell.zero_state(1, self.rnn_dtype))
    for event in prime_events[:-1]: # feed through all except last event
      x = np.zeros((1,1)) # 1x1 zero tensor
      x[0,0] = vocab[event] # encode event in this tensor
      feed = { self.input_data: x, self.initial_multicell_state: state }
      [state] = sess.run([self.final_state], feed)

    result = prime_events
    curr_event = prime_events[-1]
    for n in range(num):
      x = np.zeros((1,1)) # 1x1 zero tensor
      x[0,0] = vocab[curr_event] # encode event in this tensor
      feed = { self.input_data: x, self.initial_multicell_state: state }
      [probs, state] = sess.run([self.probs, self.final_state], feed)
      p = probs[0] # reduce 2D batch tensor to 1D distribution

      def weighted_pick(weights):
        t = np.cumsum(weights)
        s = np.sum(weights)
        return(int(np.searchsorted(t, np.random.rand(1)*s)))

      sample = weighted_pick(p)
      decoded = events[sample]
      result.append(decoded)
      curr_event = decoded

    return result



    


