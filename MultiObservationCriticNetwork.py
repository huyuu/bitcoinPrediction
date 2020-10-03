import gin
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.networks import network
from tf_agents.networks import utils
from tf_agents.networks import encoding_network

# https://www.tensorflow.org/agents/api_docs/python/tf_agents/agents/ddpg/critic_network/CriticNetwork
@gin.configurable
class MultiObservationCriticNetwork(network.Network):
  """Creates a critic network."""

  def __init__(self,
               input_tensor_spec,
               # observation_conv_layer_params=None,
               # observation_fc_layer_params=None,
               # observation_dropout_layer_params=None,
               # action_fc_layer_params=None,
               # action_dropout_layer_params=None,
               preprocessing_layers,
               preprocessing_combiner,
               joint_fc_layer_params=None,
               joint_dropout_layer_params=None,
               joint_activation_fn=tf.nn.relu,
               output_activation_fn=None,
               kernel_initializer=None,
               last_kernel_initializer=None,
               name='CriticNetwork'):
    """Creates an instance of `CriticNetwork`.
    Args:
      input_tensor_spec: A tuple of (observation, action) each a nest of
        `tensor_spec.TensorSpec` representing the inputs.
      observation_conv_layer_params: Optional list of convolution layer
        parameters for observations, where each item is a length-three tuple
        indicating (num_units, kernel_size, stride).
      observation_fc_layer_params: Optional list of fully connected parameters
        for observations, where each item is the number of units in the layer.
      observation_dropout_layer_params: Optional list of dropout layer
        parameters, each item is the fraction of input units to drop or a
        dictionary of parameters according to the keras.Dropout documentation.
        The additional parameter `permanent', if set to True, allows to apply
        dropout at inference for approximated Bayesian inference. The dropout
        layers are interleaved with the fully connected layers; there is a
        dropout layer after each fully connected layer, except if the entry in
        the list is None. This list must have the same length of
        observation_fc_layer_params, or be None.
      action_fc_layer_params: Optional list of fully connected parameters for
        actions, where each item is the number of units in the layer.
      action_dropout_layer_params: Optional list of dropout layer parameters,
        each item is the fraction of input units to drop or a dictionary of
        parameters according to the keras.Dropout documentation. The additional
        parameter `permanent', if set to True, allows to apply dropout at
        inference for approximated Bayesian inference. The dropout layers are
        interleaved with the fully connected layers; there is a dropout layer
        after each fully connected layer, except if the entry in the list is
        None. This list must have the same length of action_fc_layer_params, or
        be None.
      joint_fc_layer_params: Optional list of fully connected parameters after
        merging observations and actions, where each item is the number of units
        in the layer.
      joint_dropout_layer_params: Optional list of dropout layer parameters,
        each item is the fraction of input units to drop or a dictionary of
        parameters according to the keras.Dropout documentation. The additional
        parameter `permanent', if set to True, allows to apply dropout at
        inference for approximated Bayesian inference. The dropout layers are
        interleaved with the fully connected layers; there is a dropout layer
        after each fully connected layer, except if the entry in the list is
        None. This list must have the same length of joint_fc_layer_params, or
        be None.
      activation_fn: Activation function, e.g. tf.nn.relu, slim.leaky_relu, ...
      output_activation_fn: Activation function for the last layer. This can be
        used to restrict the range of the output. For example, one can pass
        tf.keras.activations.sigmoid here to restrict the output to be bounded
        between 0 and 1.
      kernel_initializer: kernel initializer for all layers except for the value
        regression layer. If None, a VarianceScaling initializer will be used.
      last_kernel_initializer: kernel initializer for the value regression
         layer. If None, a RandomUniform initializer will be used.
      name: A string representing name of the network.
    Raises:
      ValueError: If `observation_spec` or `action_spec` contains more than one
        observation.
    """
    super(MultiObservationCriticNetwork, self).__init__(
        input_tensor_spec=input_tensor_spec,
        state_spec=(),
        name=name)

    observation_spec, action_spec = input_tensor_spec

    flat_action_spec = tf.nest.flatten(action_spec)
    self._single_action_spec = flat_action_spec[0]
    # set up kernel_initializer
    if kernel_initializer is None:
        kernel_initializer = tf.compat.v1.keras.initializers.VarianceScaling(scale=1. / 3., mode='fan_in', distribution='uniform')
    if last_kernel_initializer is None:
        last_kernel_initializer = tf.keras.initializers.RandomUniform(minval=-0.003, maxval=0.003)
    # set up encoder_network
    self._encoder = encoding_network.EncodingNetwork(
        observation_spec,
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=preprocessing_combiner,
        conv_layer_params=None,
        fc_layer_params=None,
        dropout_layer_params=None,
        activation_fn=tf.keras.activations.relu,
        kernel_initializer=kernel_initializer,
        batch_squash=False
    )

    # TODO(kbanoop): Replace mlp_layers with encoding networks.
    # self._observation_layers = utils.mlp_layers(
    #     observation_conv_layer_params,
    #     observation_fc_layer_params,
    #     observation_dropout_layer_params,
    #     activation_fn=activation_fn,
    #     kernel_initializer=kernel_initializer,
    #     name='observation_encoding')

    # self._action_layers = utils.mlp_layers(
    #     None,
    #     action_fc_layer_params,
    #     action_dropout_layer_params,
    #     activation_fn=activation_fn,
    #     kernel_initializer=kernel_initializer,
    #     name='action_encoding')

    self._joint_layers = utils.mlp_layers(
        None,
        joint_fc_layer_params,
        joint_dropout_layer_params,
        activation_fn=joint_activation_fn,
        kernel_initializer=kernel_initializer,
        name='joint_mlp')

    self._joint_layers.append(
        tf.keras.layers.Dense(1, activation=output_activation_fn, kernel_initializer=last_kernel_initializer, name='value')
    )

  def call(self, inputs, step_type=(), network_state=(), training=False):
    observations, actions = inputs
    del step_type  # unused.

    observations, network_state = self._encoder(observations, step_type=step_type, network_state=network_state, training=training)

    # observations = tf.cast(tf.nest.flatten(observations)[0], tf.float32)
    # for layer in self._observation_layers:
    #     observations = layer(observations, training=training)

    # actions = tf.cast(tf.nest.flatten(actions)[0], tf.float32)
    # for layer in self._action_layers:
    #   actions = layer(actions, training=training)

    joint = tf.concat([observations, actions], 1)
    for layer in self._joint_layers:
        joint = layer(joint, training=training)

    return tf.reshape(joint, [-1]), network_state
