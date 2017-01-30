import tensorflow as tf # type: ignore

# dimensions
seq_length = 6
batch_size = 3
clock_size = 4
clock_states = 5
hidden_size = 7
vocab_size = 3
state_size = hidden_size - clock_size

def header(s):
  print("=" * 20 + " " + s + " " + "=" * 20)

header("dimensions")
print("--> b: batch size = ", batch_size)
print("--> s: sequence length = ", seq_length)
print("--> c: clock size = ", clock_size)
print("--> h: hidden size = ", hidden_size)
print("--> w: state size = hidden size - clock size = ", state_size)

## embeddings

vocab_embed = tf.get_variable("embedding", shape=[vocab_size, state_size])

clock_embed = tf.constant([[0,0,0,0],[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], 
    shape=[clock_states, clock_size], dtype=tf.float32)

## dummy test data

raw_clock = tf.constant([[0,1,2,3,4,1],[0,1,2,3,1,2],[0,1,2,3,1,2]],
    shape=[batch_size,seq_length])

raw_data = tf.constant([[2,1,2,1,0,1],[0,1,0,1,2,1], [0,2,0,1,0,1]],
    shape=[batch_size,seq_length])

## embedded tensors

clock_input = tf.nn.embedding_lookup(clock_embed, raw_clock)
data_input = tf.nn.embedding_lookup(vocab_embed, raw_data)

init_op = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init_op)

header("embeddings")
print("--> clock embedding:")
print(clock_embed.eval())
print("--> vocab embedding:")
print(vocab_embed.eval())

header("shapes")
print("raw_clock shape (b x s):", tf.shape(raw_clock).eval())
print(" raw_data shape (b x s):", tf.shape(raw_data).eval())
print("clock_input shape (b x s x c):", tf.shape(clock_input).eval())
print(" data_input shape (b x s x w):", tf.shape(data_input).eval())

header("raw inputs")

print("clock:")
print(raw_clock.eval())

print("data:")
print(raw_data.eval())

header("embedded tensors")
print("clock:")
print(clock_input.eval())
print("data:")
print(data_input.eval())

header("concatenated tensor")
final_input = tf.concat(2, [data_input, clock_input], name='add_clock')
print(final_input.eval())

