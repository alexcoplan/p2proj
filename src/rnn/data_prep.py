import tensorflow as tf
import tempfile

test_string = "Hello this is a test string just for example construction."


seqlen = 5

def make_example(eg_string):
  ex = tf.train.SequenceExample()

  # context features apply to the entire example
  ex.context.feature["length"].int64_list.value.append(seqlen)

  fl_chars = ex.feature_lists.feature_list["chars"]
  for c in eg_string:
    fl_chars..feature.add().int32_list.value.append(ord(c))

  return ex

with tempfile.NamedTemporaryFile() as fp:
  writer = tf.python_io.TFRecordWriter(fp.name)
  for sequence
