import tensorflow as tf
import math

def mlp(input_x, image_pixels=28, hidden_units=100, num_classes=10):
  hid_w = tf.Variable(tf.truncated_normal([image_pixels * image_pixels, hidden_units],stddev=1.0 / image_pixels), name='hid_w')
  hid_b = tf.Variable(tf.zeros([hidden_units]), name='hid_b')

  sm_w = tf.Variable(tf.truncated_normal([hidden_units, 10], stddev=1.0 / math.sqrt(hidden_units)), name='sm_w')
  sm_b = tf.Variable(tf.zeros([num_classes]), name='sm_b')

  hid_lin = tf.nn.xw_plus_b(input_x, hid_w, hid_b)
  hid = tf.nn.relu(hid_lin)

  logits = tf.add(tf.matmul(hid, sm_w), sm_b)
  return logits


