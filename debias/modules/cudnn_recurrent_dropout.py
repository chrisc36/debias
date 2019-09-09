import tensorflow as tf
from tensorflow.contrib.cudnn_rnn.python import layers as cudnn_layers
from tensorflow.contrib.cudnn_rnn.python.ops import cudnn_rnn_ops
from tensorflow.python import TruncatedNormal

from debias.modules.layers import SequenceMapper
from debias.utils import ops


class CudnnLSTMRecurrentDropout(SequenceMapper):
  """CudnnLSTM with hacked in recurrent dropped

  The dropout is done by dropping out particular columns from the hidden weight matrix
  and then recaling, which is equivalent to dropping out hidden units. This means
  the effective dropout-mask is shared both between all examples in the batch, but in practice
  that does not seem to be an issue, and in exchange we get to use the super fast
  Cudnn LSTM implementation with recurrent dropout, which can be pretty impactful
  """
  def __init__(self, n_out: int, dropout: float, learn_initial_states: bool=True,
               direction="bi", lstm_bias=1):
    self.dropout = dropout
    self.lstm_bias = lstm_bias
    self.learn_initial_states = learn_initial_states
    self.n_out = n_out
    self.direction = direction

  def _apply_transposed(self, is_train, x, initial_states=None):
    w_init = TruncatedNormal(stddev=0.05)
    x_size = x.shape.as_list()[-1]
    if x_size is None:
      raise ValueError("Last dimension must be defined (have shape %s)" % str(x.shape))

    cell = cudnn_rnn_ops.CudnnLSTM(1, self.n_out, x_size, input_mode="linear_input")

    # We need to know the mapping of weights/baises -> CudnnLSTM parameter, so just
    # build a `CudnnLSTM` and read its fields
    c = cudnn_layers.CudnnLSTM(1, self.n_out)
    c._input_size = x.shape.as_list()[-1]
    w_shapes = c.canonical_weight_shapes
    b_shapes = c.canonical_bias_shapes
    weights = [w_init(s, tf.float32) for s in w_shapes]
    biases = [tf.zeros(s, tf.float32) for s in b_shapes]
    biases[1] = tf.constant(self.lstm_bias / 2.0, tf.float32, b_shapes[1])
    biases[5] = tf.constant(self.lstm_bias / 2.0, tf.float32, b_shapes[5])

    opaque_params_t = cell.canonical_to_params(weights, biases)
    parameters = tf.get_variable("opaque_kernel", initializer=opaque_params_t, validate_shape=False)

    p = 1.0 - self.dropout

    if is_train and self.dropout > 0:
      mult_bias = [tf.ones_like(x) for x in biases]
      mult_w = [tf.ones_like(x) for x in weights]

      bias_mask = tf.floor(tf.random_uniform((self.n_out,), p, 1 + p)) / p

      for j in range(4, 8):
        mult_w[j] *= tf.expand_dims(bias_mask, 0)

      mult_mask = cell.canonical_to_params(mult_w, mult_bias)
      parameters = parameters * mult_mask

    initial_state_h, initial_state_c = initial_states
    out = cell(x, initial_state_h, initial_state_c, parameters, True)[0]

    return out

  def apply(self, is_train, x, mask=None, inital_states=None):
    x_t = tf.transpose(x, [1, 0, 2])
    batch = ops.get_shape_tuple(x_t, 1)
    bidr = self.direction == "bi"

    if inital_states is not None:
      inital_states = tuple(tf.expand_dims(x, 0) for x in inital_states)

    if self.learn_initial_states and inital_states is None:
      if bidr:
        names = ["fw_h", "fw_c", "bw_h", "bw_c"]
      else:
        names = ["fw_h", "fw_c"]

      initial_states = []
      for n in names:
        v = tf.get_variable(n, (1, self.n_out), tf.float32, tf.zeros_initializer())
        initial_states.append(tf.tile(tf.expand_dims(v, 1), [1, batch, 1]))
    else:
      initial_states = [tf.zeros((1, batch, self.n_out)) for _ in range(2 + 2 * bidr)]

    if self.direction == 'bi':
      with tf.variable_scope("forward"):
        fw = self._apply_transposed(is_train, x_t, initial_states=initial_states[:2])
      with tf.variable_scope("backward"):
        x_bw = x_t[::-1] if mask is None else tf.reverse_sequence(x_t, mask, 0, 1)
        bw = self._apply_transposed(is_train, x_bw, initial_states=initial_states[2:])
        bw = bw[::-1] if mask is None else tf.reverse_sequence(bw, mask, 0, 1)
      out = tf.concat([fw, bw], axis=2)
    elif self.direction == "fw":
      out = self._apply_transposed(is_train, x_t, initial_states=initial_states)
    elif self.direction == "bw":
      x_bw = x_t[::-1] if mask is None else tf.reverse_sequence(x_t, mask, 0, 1)
      bw = self._apply_transposed(is_train, x_bw, initial_states=initial_states)
      out = bw[::-1] if mask is None else tf.reverse_sequence(bw, mask, 0, 1)
    else:
      raise ValueError()

    out = tf.transpose(out, [1, 0, 2])

    if mask is not None:
      out *= tf.expand_dims(tf.cast(tf.sequence_mask(mask, tf.shape(out)[1]), tf.float32), 2)
    return out

