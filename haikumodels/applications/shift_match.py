from typing import Any, Callable, Iterator, Mapping, Optional, Sequence, Union

import h5py
import haiku as hk
import jax
import jax.numpy as jnp
from jax import vmap
from jax import lax

from .. import utils
from ..ops import BatchNorm

hk.BatchNorm = BatchNorm

BASE_URL = "https://github.com/abarcel/haikumodels/releases/download/v0.1/"


class ShiftMatch(hk.Module):
  def __init__(self, match_type, name="shift_match"):
    assert match_type in [
        'channel_wise_sep_cov_mean'
    ]
    self.match_type = match_type
    self.sqrt_cov_h_train = None
    super().__init__(name=name)

  def _reshape(self, x, match_type, mode):
    '''
    Assume x is of shape:
    (N, H, W, C)
    '''
    N, H, W, C = x.shape
    batch_size = x.shape[0]
    if mode == 'match':
        return x.reshape(C, N * W, H)
    x_H = x.transpose(3, 0, 2, 1).reshape(C, N * W, H)
    x_W = x.transpose(3, 0, 1, 2).reshape(C, N * H, W)
    return jnp.stack([x_H, x_W])

  def _matrix_sqrt(self, cov, neg=False):
    if neg:
      cov = cov * (jnp.ones_like(cov) + jnp.eye(cov.shape[-1]) * 1e-3)
    e, v = lax.linalg.eigh(cov)
    if neg:
      # print(v.min())
      # print(v.max())
      # print(jnp.diag(cov).mean())
      EPS=1e-8
    else:
      EPS=1e-8
    v = jnp.maximum(v, jnp.ones_like(v) * EPS)
    v = jnp.diag(v) if not neg else jnp.diag(1/v)
    return e @ v ** 0.5 @ e.T

  def _batch_outer_product(self, x):
    assert len(x.shape) == 2
    return vmap(lambda _x: jnp.outer(_x, _x))(x)

  def _fft_match(self, h_test, spec_train_sqr):
    fft_test = []
    batch_num = 4
    batch_size = h_test.shape[0] // batch_num
    h_test = h_test.reshape((batch_num, batch_size, *h_test.shape[1:]))
    for i in range(batch_num):
      fft_test.append(jnp.fft.fft2(h_test[i]))
    fft_test = jnp.concatenate(fft_test, 0)
    # fft_test = jnp.fft.fft2(h_test)
    spec_test = jnp.sqrt(jnp.mean(jnp.abs(fft_test)**2, (0))) + 1E-10
    matched_fft_feature = (fft_test / spec_test) * jnp.sqrt(spec_train_sqr)
    ifft_test = []
    matched_fft_feature = matched_fft_feature.reshape((batch_num, batch_size,
                  *matched_fft_feature.shape[1:]))
    for i in range(batch_num):
      ifft_test.append(jnp.fft.ifft2(matched_fft_feature[i]))
    out = jnp.real(jnp.concatenate(ifft_test, 0))
    return out


  def _match(self, h_test, cov_x_train):
    n_devices = jax.local_device_count()
    match_ind_channel = 'wise' in self.match_type
    def _inner(h_test, sqrt_cov_h_train):
      cov_h_test = h_test.T @ h_test / len(h_test)
      neg_sqrt_cov_h_test = self._matrix_sqrt(cov_h_test, neg=True)
      if n_devices > 1:
        out = pmap(
          lambda h: h @ neg_sqrt_cov_h_test @ sqrt_cov_h_train
          )(batch_split_axis(h_test, n_devices))
        return out.reshape(
            (out.shape[0] * out.shape[1],) + out.shape[2:]
            )
      # print(h_test.shape)
      # print(neg_sqrt_cov_h_test.shape)
      # print(sqrt_cov_h_train.shape)
      return h_test @ neg_sqrt_cov_h_test @ sqrt_cov_h_train
    if self.sqrt_cov_h_train:
      # Cache the sqrt of the training covariance matrix.
      sqrt_cov_h_train = self.sqrt_cov_h_train
    else:
      if match_ind_channel:
        sqrt_cov_h_train = vmap(self._matrix_sqrt)(cov_x_train)
        self.sqrt_cov_h_train = sqrt_cov_h_train
      else:
        sqrt_cov_h_train = self._matrix_sqrt(cov_x_train)
        self.sqrt_cov_h_train = sqrt_cov_h_train

    match_func = vmap(_inner) if match_ind_channel else _inner
    return match_func(h_test, self.sqrt_cov_h_train)

  def __call__(self, x, mode=None):
    '''
    Entry point for shift_match invoke.
    '''
    if self.match_type == 'None':
      return x
    assert len(x.shape) == 4 # This module currently only supports CNN feature.
    _x = self._reshape(x, self.match_type, mode)
    D = x.shape[-2]
    C = x.shape[-1]
    old_shape = x.shape
    # Initialize the covariance matrix.
    cov_counter = hk.get_state("counter", shape=[], dtype=jnp.int32, init=jnp.zeros)
    cov_H = hk.get_state('cov_H', shape=(C, D, D), dtype=jnp.float32, init=jnp.zeros)
    mu_H = hk.get_state('mu_H', shape=(C, D), dtype=jnp.float32, init=jnp.zeros)
    cov_W = hk.get_state('cov_W', shape=(C, D, D), dtype=jnp.float32, init=jnp.zeros)
    mu_W = hk.get_state('mu_W', shape=(C, D), dtype=jnp.float32, init=jnp.zeros)
    cov = (cov_H, cov_W)
    mu = (mu_H, mu_W)
    if not mode:
      return x
    elif mode == 'acc':
        cov_H, cov_W = cov
        mu_H, mu_W = mu
        x_H, x_W = _x
        batch_size = x_H.shape[1]
        new_cov_H = vmap(lambda x: x.T @ x)(x_H)
        new_mu_H = x_H.sum(1)
        new_cov_W = vmap(lambda x: x.T @ x)(x_W)
        new_mu_W = x_W.sum(1)
        hk.set_state('cov_H', cov_H + new_cov_H)
        hk.set_state('mu_H', mu_H + (new_mu_H - mu_H * batch_size) / (cov_counter + batch_size))
        hk.set_state('cov_W', cov_W + new_cov_W)
        hk.set_state('mu_W', mu_W + (new_mu_W - mu_W * batch_size) / (cov_counter + batch_size))
        hk.set_state('counter', cov_counter + x_H.shape[1])
        return x
    elif mode == 'match':
      if not self.match_type:
        return x
      elif self.match_type == 'channel_wise_sep_cov_mean':
        N, H, W, C = old_shape
        R_H, R_W = cov # Second moment
        mu_H, mu_W = mu
        x = x.transpose(3, 0, 2, 1).reshape(C, N * W, H)
        mu_test = x.mean(1, keepdims=True)
        x = self._match(x - mu_test,
            R_H / cov_counter - self._batch_outer_product(mu_H)) + mu_H.reshape(mu_test.shape)
        x = x.reshape(C, N, W, H).transpose(0, 1, 3, 2)
        self.sqrt_cov_h_train = None
        x = x.reshape(C, N * H, W)
        mu_test = x.mean(1, keepdims=True)
        x = self._match(x - mu_test,
        R_W / cov_counter - self._batch_outer_product(mu_W)) + mu_W.reshape(mu_test.shape)
        x = x.reshape(C, N, H, W).transpose(1, 2, 3, 0)
        return x
    return


def shift_match_builder(match_mode='channel_wise_sep_cov_mean'):
  return ShiftMatch(match_mode)



class block1(hk.Module):

  def __init__(
      self,
      output_channels: int,
      conv_shortcut: bool = True,
      kernel_shape: int = 3,
      stride: int = 1,
      weights_init: Iterator[Any] = None,
      wb_init: Mapping[str, Callable[[Sequence[int], Any], jnp.ndarray]] = None,
      bn_config: Mapping[str, Union[str, float, bool]] = None,
      name: Optional[str] = None,
  ):

    super().__init__(name=name)
    self.conv_shortcut = conv_shortcut

    if conv_shortcut is True:
      self.shortcut_conv = hk.Conv2D(
          output_channels=4 * output_channels,
          kernel_shape=1,
          stride=stride,
          padding="VALID",
          name="conv_shortcut",
          **next(weights_init),
          **wb_init,
      )
      self.shortcut_conv_bn = hk.BatchNorm(
          name="conv_shortcut_bn",
          **next(weights_init),
          **bn_config,
      )

    self.conv1 = hk.Conv2D(
        output_channels=output_channels,
        kernel_shape=1,
        stride=stride,
        padding="VALID",
        name="conv1",
        **next(weights_init),
        **wb_init,
    )
    self.conv1_bn = hk.BatchNorm(
        name="conv1_bn",
        **next(weights_init),
        **bn_config,
    )

    self.conv2 = hk.Conv2D(
        output_channels=output_channels,
        kernel_shape=kernel_shape,
        name="conv2",
        **next(weights_init),
        **wb_init,
    )
    self.conv2_bn = hk.BatchNorm(
        name="conv2_bn",
        **next(weights_init),
        **bn_config,
    )

    self.conv3 = hk.Conv2D(
        output_channels=4 * output_channels,
        kernel_shape=1,
        padding="VALID",
        name="conv3",
        **next(weights_init),
        **wb_init,
    )
    self.conv3_bn = hk.BatchNorm(
        name="conv3_bn",
        **next(weights_init),
        **bn_config,
    )

  def __call__(self, inputs: jnp.ndarray, is_training: bool, sm_mode):
    out = shortcut = inputs

    if self.conv_shortcut is True:
      shortcut = self.shortcut_conv(inputs)
      shortcut = self.shortcut_conv_bn(shortcut, is_training)

    out = self.conv1(out)
    out = self.conv1_bn(out, is_training)
    out = shift_match_builder()(out, sm_mode)
    out = jax.nn.relu(out)
    out = self.conv2(out)
    out = self.conv2_bn(out, is_training)
    out = shift_match_builder()(out, sm_mode)
    out = jax.nn.relu(out)
    out = self.conv3(out)
    out = self.conv3_bn(out, is_training)
    out = shift_match_builder()(out, sm_mode)
    out = jax.nn.relu(out + shortcut)

    return out


class stack1(hk.Module):

  def __init__(
      self,
      output_channels: int,
      blocks: int,
      stride1: int = 2,
      weights_init: Iterator[Any] = None,
      wb_init: Mapping[str, Callable[[Sequence[int], Any], jnp.ndarray]] = None,
      bn_config: Mapping[str, Union[str, float, bool]] = None,
      name: Optional[str] = None,
  ):

    super().__init__(name=name)

    self.blocks = []

    self.blocks.append(
        block1(
            output_channels=output_channels,
            stride=stride1,
            weights_init=weights_init,
            wb_init=wb_init,
            bn_config=bn_config,
            name="block01",
        ))

    for i in range(2, blocks + 1):
      self.blocks.append(
          block1(
              output_channels=output_channels,
              conv_shortcut=False,
              weights_init=weights_init,
              wb_init=wb_init,
              bn_config=bn_config,
              name=f"block{i:02d}",
          ))

  def __call__(self, inputs: jnp.ndarray, is_training: bool, sm_mode):
    out = inputs

    for block in self.blocks:
      out = block(out, is_training, sm_mode)

    return out


class ResNet(hk.Module):
  """Instantiates the ResNet architecture.
    See https://arxiv.org/pdf/1512.03385.pdf for details.
    Optionally loads weights pre-trained on ImageNet.
    Note that the default input image size for this model is 224x224.
    """

  CONFIGS = {
      "ResNet50": {
          "blocks_per_group": (3, 4, 6, 3),
          "channels_per_group": (64, 128, 256, 512),
          "strides_per_group": (1, 2, 2, 2),
      },
  }

  def __init__(
      self,
      blocks_per_group: Sequence[int],
      channels_per_group: Sequence[int],
      strides_per_group: Sequence[int],
      include_top: bool = True,
      weights: str = "imagenet",
      pooling: Optional[str] = None,
      classes: int = 1000,
      classifier_activation: Callable[[jnp.ndarray],
                                      jnp.ndarray] = jax.nn.softmax,
      wb_init: Mapping[str, Callable[[Sequence[int], Any], jnp.ndarray]] = None,
      bn_config: Mapping[str, Union[str, float, bool]] = None,
      ckpt_dir: Optional[str] = None,
      name: Optional[str] = None,
  ):
    """Initializes a ResNet function.
        Args:
          blocks_per_group: A sequence of length 4 that indicates the number of
            blocks created in each group.
          channels_per_group: A sequence of length 4 that indicates the channel size
            for blocks created in each group.
          strides_per_group: A sequence of length 4 that indicates the stride
            for blocks created in each group.
          include_top: Whether to include the fully-connected layer at the top
            of the network.
            By default, True.
          weights: One of None (random initialization) or ``imagenet``
            (pre-trained on ImageNet).
            By default, ``imagenet``.
          pooling: Pooling for feature extraction when `include_top` is False.
            (`pooling`, when `include_top` is True, defaults to ``avg`` and
            changes to `pooling` will be uneffective):
            - None means that the output of the model will be the 4D tensor
              output of the last convolutional block.
            - ``avg`` means that global average pooling will be applied to the
              output of the last convolutional block, and thus the output of
              the model will be a 2D tensor.
            - ``max`` means that global max pooling will be applied.
            By default, when `include_top` is False, `pooling` is None.
          classes: Number of classes to classify images into,
            only to be optionally specified if `include_top` is True
            and `weights` argument is None.
            By default, ``1000``.
          classifier_activation: A ``jax.nn`` activation function to use on the
            "top" layer. Ignored unless `include_top` is True.
            Set `classifier_activation` to None to return the logits of the
            "top" layer. When `weights` is ``imagenet``,
            `classifier_activation` can only be set to None or ``jax.nn.softmax``.
            By default, ``jax.nn.softmax``.
          wb_init: Dictionary of two elements, ``w_init`` and ``b_init``
            weight initializers for both dense layer and convolution layers.
            Optionally specified only when `weights` is None.
            By default, ``w_init`` is truncated normal and ``b_init`` is zeros.
          bn_config: Dictionary of two elements, ``decay_rate`` and
            ``eps`` to be passed on to the :class:``~haiku.BatchNorm`` layers.
            By default, ``decay_rate`` is ``0.99`` and ``eps`` is ``1e-5``.
          ckpt_dir: Optional path to download pretrained weights.
            By default, temporary system file directory.
          name: Optional name name of the module.
        """

    super().__init__(name=name)
    self.stack_groups = []
    self.default_size = 224
    self.min_size = 32
    self.blocks_per_group = blocks_per_group
    self.channels_per_group = channels_per_group
    self.strides_per_group = strides_per_group
    self.include_top = include_top
    self.weights = weights
    self.pooling = pooling
    self.classes = 1000 if weights == "imagenet" else classes
    self.classifier_activation = classifier_activation
    self.ckpt_dir = ckpt_dir
    self.name = name

    if weights == "imagenet":
      wb_init, bn_config = None, None

    self.wb_init = dict(wb_init or {})

    self.bn_config = dict(bn_config or {})
    self.bn_config.setdefault("decay_rate", 0.99)
    self.bn_config.setdefault("eps", 1e-5)
    self.bn_config.setdefault("create_scale", True)
    self.bn_config.setdefault("create_offset", True)

  def init_stacks(self, inputs: jnp.ndarray):
    if self.weights == "imagenet" and self.include_top:
      if self.classes != 1000:
        print("When setting `include_top` as True "
              "and loading from ``imagenet`` weights, "
              "`classes` must be ``1000``."
              "\tEntered value ``" + str(self.classes) +
              "`` is replaced with ``1000``.")
        self.classes = 1000
      if self.classifier_activation is not (None or jax.nn.softmax):
        print("When setting `include_top` as True and loading "
              "from ``imagenet`` weights, `classifier_activation` "
              "must be None or ``jax.nn.softmax``."
              "\tEntered setting is replaced with ``jax.nn.softmax``.")
        self.classifier_activation = jax.nn.softmax
      if inputs.shape[1:] != (self.default_size, self.default_size, 3):
        raise ValueError("When setting `include_top` as True "
                         "and loading ``imagenet`` weights, "
                         "`inputs` shape should be " +
                         str((None, self.default_size, self.default_size, 3)) +
                         " where None can be any natural number.")
    if (inputs.shape[1] < self.min_size) or (inputs.shape[2] < self.min_size):
      raise ValueError("Input size must be at least " + str(self.min_size) +
                       "x" + str(self.min_size) + "; got `inputs` shape as ``" +
                       str(inputs.shape[1:3]) + "``.")

    model_weights = None
    if self.weights == "imagenet":
      file_name = self.name + "_weights_tf_dim_ordering_tf_kernels.h5"
      ckpt_file = utils.download(self.ckpt_dir, BASE_URL + file_name)
      model_weights = h5py.File(ckpt_file, "r")
    weights_init = utils.load_weights(model_weights)

    stack = stack1

    self.conv1 = hk.Conv2D(
        output_channels=64,
        kernel_shape=7,
        stride=2,
        padding="VALID",
        name="group01_conv1",
        **next(weights_init),
        **self.wb_init,
    )

    self.conv1_bn = hk.BatchNorm(
          name="group01_conv1_bn",
          **next(weights_init),
          **self.bn_config,
      )

    for i in range(4):
      self.stack_groups.append(
          stack(
              output_channels=self.channels_per_group[i],
              blocks=self.blocks_per_group[i],
              stride1=self.strides_per_group[i],
              weights_init=weights_init,
              wb_init=self.wb_init,
              bn_config=self.bn_config,
              name=f"group{i+2:02d}",
          ))

    if self.include_top:
      self.top_layer = hk.Linear(
          output_size=self.classes,
          name="top_layer",
          **next(weights_init),
          **self.wb_init,
      )

  def __call__(self, inputs: jnp.ndarray, is_training: bool, sm_mode=None,
              exclude_input=False, input_only=False):
    out = inputs

    if not self.stack_groups:
      self.init_stacks(inputs)
    if exclude_input:
      out = shift_match_builder()(out, None)
    else:
      out = shift_match_builder()(out, sm_mode)

    if input_only:
      sm_mode = None
    
    out = jnp.pad(out, ((0, 0), (3, 3), (3, 3), (0, 0)),
                  "constant",
                  constant_values=(0, 0))
    # sm_mode = None
    out = self.conv1(out)
    out = self.conv1_bn(out, is_training)
    out = shift_match_builder()(out, sm_mode)
    # out = shift_match_builder()(out, None)
    out = jax.nn.relu(out)
    out = jnp.pad(out, ((0, 0), (1, 1), (1, 1), (0, 0)),
                  "constant",
                  constant_values=(0, 0))
    out = hk.max_pool(out, window_shape=3, strides=2, padding="VALID")

    for stack_group in self.stack_groups:
      out = stack_group(out, is_training, sm_mode)

    if self.include_top:
      out = jnp.mean(out, axis=(1, 2))
      out = self.top_layer(out)
      if self.classifier_activation:
        out = self.classifier_activation(out)
    else:
      if self.pooling == "avg":
        out = jnp.mean(out, axis=(1, 2))
      elif self.pooling == "max":
        out = jnp.max(out, axis=(1, 2))

    return out

  def extract_first_feature(self, inputs: jnp.ndarray, is_training: bool, sm_mode=None,
              exclude_input=False, input_only=False):
    out = inputs

    if not self.stack_groups:
      self.init_stacks(inputs)
    if exclude_input:
      out = shift_match_builder()(out, None)
    else:
      out = shift_match_builder()(out, sm_mode)

    if input_only:
      sm_mode = None
    
    out = jnp.pad(out, ((0, 0), (3, 3), (3, 3), (0, 0)),
                  "constant",
                  constant_values=(0, 0))
    # sm_mode = None
    out = self.conv1(out)
    out = self.conv1_bn(out, is_training)
    return out    


class ResNet50(ResNet):
  """Instantiates the ResNet50 architecture.
    NOTE: Information about Args can be found in main ResNet
    """

  def __init__(
      self,
      include_top: bool = True,
      weights: str = "imagenet",
      pooling: Optional[str] = None,
      classes: int = 1000,
      classifier_activation: Callable[[jnp.ndarray],
                                      jnp.ndarray] = jax.nn.softmax,
      wb_init: Mapping[str, Callable[[Sequence[int], Any], jnp.ndarray]] = None,
      bn_config: Mapping[str, Union[str, float, bool]] = None,
      ckpt_dir: Optional[str] = None,
  ):
    super().__init__(
        include_top=include_top,
        weights=weights,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        wb_init=wb_init,
        bn_config=bn_config,
        ckpt_dir=ckpt_dir,
        name="resnet50",
        **ResNet.CONFIGS["ResNet50"],
    )

def preprocess_input(x):
  return utils.preprocess_input(x, mode="caffe")
