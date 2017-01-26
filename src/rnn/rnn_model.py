import tensorflow as tf # type: ignore
import numpy as np # type: ignore

from enum import Enum

class SamplingMethod(Enum):
  WEIGHTED_PICK = 0
  HARD_MAX = 1

class ModelConfig(object):
  def __init__(self, data_loader):
    self.keep_prob = 0.5
    self.max_grad_norm = 5
    self.learning_rate = 1.0
    self.num_epochs = 100
    self.hidden_size = 256
    self.lr_decay = 0.98
    self.num_layers = 2
    self.mode       = data_loader.mode
    self.batch_size = data_loader.batch_size
    self.num_test_examples = data_loader.num_test_examples
    self.seq_length = data_loader.seq_length
    self.vocab_size = data_loader.vocab_size

class Model(object):
  rnn_dtype = tf.float32

  def __init__(self, config, is_training=True):
    # training hyperparams
    vocab_size = config.vocab_size
    seq_length = config.seq_length
    hidden_units = config.hidden_size # width of LSTM hidden layer

    # if we're not training, set up the RNN for sampling
    if not is_training:
      seq_length = 1

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_units, state_is_tuple=True)
    if is_training and config.keep_prob < 1.0 and config.num_layers > 1:
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
    [vocab_size, hidden_units], dtype=self.rnn_dtype)
    inputs = tf.nn.embedding_lookup(embedding, self.input_data)

    # could add dropout here... see Zaremba et al. 2014

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

    # transform the outputs back into the input matrix form
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

    if not is_training:
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

  def sample(self, sess, events, vocab, num, prime_events,
    method=SamplingMethod.WEIGHTED_PICK):
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

      if method == SamplingMethod.WEIGHTED_PICK:
        sample = weighted_pick(p)
      elif method == SamplingMethod.HARD_MAX:
        sample = np.argmax(p)
      else:
        raise NotImplementedError("Bad sampling type")

      decoded = events[sample]
      result.append(decoded)
      curr_event = decoded

    return result



    


