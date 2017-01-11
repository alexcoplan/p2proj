import tensorflow as tf

import numpy as np

class ModelConfig(object):
  max_grad_norm = 5
  learning_rate = 1.0
  num_epochs = 20
  hidden_size = 128
  lr_decay = 0.8

  def __init__(self, text_loader):
    self.batch_size = text_loader.batch_size
    self.seq_length = text_loader.seq_length
    self.vocab_size = text_loader.vocab_size

class Model(object):
  def __init__(self, config, is_training=True):
    # training hyperparams
    vocab_size = config.vocab_size
    batch_size = config.batch_size
    seq_length = config.seq_length
    hidden_units = config.hidden_size # width of LSTM hidden layer
    param_dtype = tf.float32

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_units, state_is_tuple=True)
    multi_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell], state_is_tuple=True)

    self.input_data = tf.placeholder(tf.int32, [batch_size, seq_length])
    self.target_data = tf.placeholder(tf.int32, [batch_size, seq_length])

    # get initial state for our LSTM cell
    self.initial_multicell_state = multi_cell.zero_state(batch_size, param_dtype)

    # might need to add with tf.device("/cpu:0") here if training on GPU
    # create embedding variable which is randomly initialised
    # the network will learn a dense representation for the input vocabulary
    embedding = tf.get_variable("embedding", 
    [vocab_size, hidden_units], dtype=param_dtype)
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
    outputs, state = tf.nn.rnn(multi_cell, inputs, dtype=param_dtype)

    # transform the outputs back into the input matrix form
    output = tf.reshape(tf.concat(1, outputs), [-1, hidden_units])
    softmax_w = tf.get_variable("softmax_w", [hidden_units, vocab_size])
    softmax_b = tf.get_variable("softmax_b", [vocab_size])
    logits = tf.matmul(output, softmax_w) + softmax_b
    loss = tf.nn.seq2seq.sequence_loss_by_example(
        [logits],
        [tf.reshape(self.target_data, [-1])],
        [tf.ones([batch_size * seq_length], dtype=param_dtype)],
        vocab_size
    )
    # nb we are doing truncated backpropagation with the truncation point
    # set to seq_length

    self.cost = tf.reduce_sum(loss) / batch_size
    self.final_state = state

    if not is_training:
      return

    self.lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
        config.max_grad_norm)
    
    optimizer = tf.train.GradientDescentOptimizer(self.lr)
    self.train_op = optimizer.apply_gradients(zip(grads, tvars))
    
    self.new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
    self.lr_update = tf.assign(self.lr, self.new_lr)

  def assign_lr(self, sess, lr_value):
    sess.run(self.lr_update, feed_dict={self.new_lr: lr_value})


    

    


