# Importing
# Python Modules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as nu
import pandas as pd
import datetime as dt
import multiprocessing as mp
from matplotlib import pyplot as pl
import pickle
import sys
import os
# Tensorflow Modules
import tensorflow as tf
tf.compat.v1.enable_v2_behavior()
from tensorflow import keras as kr
from tf_agents.networks import encoding_network, utils
from tf_agents.networks.network import Network
from tf_agents.networks.q_network import QNetwork
from tf_agents.networks.value_network import ValueNetwork
from tf_agents.agents.ddpg import critic_network
from tf_agents.agents import DdpgAgent
from tf_agents.agents.sac import sac_agent, tanh_normal_projection_network
from tf_agents.drivers import dynamic_step_driver, dynamic_episode_driver
from tf_agents.environments import tf_py_environment
from tf_agents.experimental.train import actor, learner, triggers
from tf_agents.experimental.train.utils import spec_utils, strategy_utils, train_utils
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_network, normal_projection_network, value_network, q_network
from tf_agents.policies import greedy_policy, random_tf_policy, policy_saver
from tf_agents.replay_buffers import tf_uniform_replay_buffer, reverb_replay_buffer, reverb_utils
from tf_agents.trajectories import trajectory
from tf_agents.utils import common, nest_utils
import shutil
# Custom Modules
from BitcoinEnvironment import BTC_JPY_Environment
from MultiObservationCriticNetwork import MultiObservationCriticNetwork

# Model
# https://www.tensorflow.org/agents/tutorials/8_networks_tutorial?hl=en
class CustomActorNetwork(Network):
    def __init__(self,
            observation_spec,
            action_spec,
            preprocessing_layers=None,
            preprocessing_combiner=None,
            conv_layer_params=None,
            fc_layer_params=(75, 40),
            dropout_layer_params=None,
            # enable_last_layer_zero_initializer=False,
            name='ActorNetwork'):
        # call super
        super(CustomActorNetwork, self).__init__(input_tensor_spec=observation_spec, state_spec=(), name=name)
        # check action_spec
        self._action_spec = action_spec
        flat_action_spec = tf.nest.flatten(action_spec)
        if len(flat_action_spec) != 1:
            raise ValueError('flatten action_spec should be len=2, but get len={}'.format(len(flat_action_spec)))
        self._single_action_spec = flat_action_spec[0]
        # set up kernel_initializer
        # kernel_initializer = tf.keras.initializers.VarianceScaling(scale=1. / 3., mode='fan_in', distribution='uniform')
        # set up encoder_network
        self._encoder = encoding_network.EncodingNetwork(
            observation_spec,
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            conv_layer_params=conv_layer_params,
            fc_layer_params=fc_layer_params,
            dropout_layer_params=dropout_layer_params,
            activation_fn=tf.keras.activations.relu,
            # kernel_initializer=kernel_initializer,
            batch_squash=False
        )
        # set up action_projection layer
        # initializer = tf.keras.initializers.RandomUniform(minval=-0.003, maxval=0.003)
        self._action_projection_layer = tf.keras.layers.Dense(
            flat_action_spec[0].shape.num_elements(),
            activation=tf.keras.activations.tanh,
            # kernel_initializer=initializer,
            name='action_projection_layer'
        )


    def call(self, observations, step_type=(), network_state=()):
        outer_rank = nest_utils.get_outer_rank(observations, self.input_tensor_spec)
        # We use batch_squash here in case the observations have a time sequence
        # compoment.
        batch_squash = utils.BatchSquash(outer_rank)
        observations = tf.nest.map_structure(batch_squash.flatten, observations)

        state, network_state = self._encoder(observations, step_type=step_type, network_state=network_state)
        actions = self._action_projection_layer(state)
        actions = common.scale_to_spec(actions, self._single_action_spec)
        actions = batch_squash.unflatten(actions)
        return tf.nest.pack_sequence_as(self._action_spec, [actions]), network_state



if __name__ == '__main__':
    mp.freeze_support()
    # check if should continue from last stored checkpoint
    if len(sys.argv) == 2:
        shouldContinueFromLastCheckpoint = sys.argv[1] == '-c'
    else:
        shouldContinueFromLastCheckpoint = False
    # Environment

    # create environment and transfer it to Tensorflow version
    print('Creating environment ...')
    gamma = 0.999
    env = BTC_JPY_Environment(imageWidth=int(24*4), imageHeight=int(24*8), initialAsset=100000, isHugeMemorryMode=True, shouldGiveRewardsFinally=True, gamma=gamma)
    episodeEndSteps = env.episodeEndSteps
    env = tf_py_environment.TFPyEnvironment(env)
    evaluate_env = tf_py_environment.TFPyEnvironment(BTC_JPY_Environment(imageWidth=int(24*4), imageHeight=int(24*8), initialAsset=100000, isHugeMemorryMode=False, shouldGiveRewardsFinally=True, gamma=gamma))
    observation_spec = env.observation_spec()
    action_spec = env.action_spec()
    print('Environment created.')

    # Hyperparameters
    criticLearningRate = 1e-6
    actorLearningRate = 1e-6

    batchSize = 1
    target_update_tau = 1e-4

    critic_observationDenseLayerParams = [int(observation_spec['observation_market'].shape[0]//100)]
    critic_commonDenseLayerParams = [int(observation_spec['observation_market'].shape[0]//100)]
    # actor_convLayerParams = [(96, 3, 1), (24, 3, 1)]
    actor_denseLayerParams = [int(observation_spec['observation_market'].shape[0]//100)]

    num_iterations = int(1e6)
    log_interval = num_iterations//1000
    eval_interval = num_iterations//100
    collect_episodes_per_iteration = 10
    _storeFullEpisodes = 100
    replayBufferCapacity = int(_storeFullEpisodes * episodeEndSteps * batchSize)
    warmupEpisodes = _storeFullEpisodes
    validateEpisodes = 10

    checkpointDir = './DDPGAgent_checkcpoints'
    if not os.path.exists(checkpointDir):
        os.mkdir(checkpointDir)

    policyDir = './DDPGAgent_savedPolicy'
    if not os.path.exists(policyDir):
        os.mkdir(policyDir)


    # Actor
    actor_net = CustomActorNetwork(
        observation_spec,
        action_spec,
        preprocessing_layers={
            'observation_market': kr.models.Sequential([
                kr.layers.Conv2D(filters=int((observation_spec['observation_market'].shape[0]*observation_spec['observation_market'].shape[1])//100), kernel_size=3, activation='relu', input_shape=(observation_spec['observation_market'].shape[0], observation_spec['observation_market'].shape[1], 1)),
                # kr.layers.Conv2D(filters=int((observation_spec[0].shape[0]*observation_spec[0].shape[1])//8), kernel_size=3, activation='relu', input_shape=(observation_spec[0].shape[0], observation_spec[0].shape[1], 1)),
                kr.layers.Flatten()
            ]),
            # 'observation_market': kr.layers.Conv2D(filters=int((observation_spec['observation_market'].shape[0]*observation_spec['observation_market'].shape[1])//100), kernel_size=3, activation='relu', input_shape=(observation_spec['observation_market'].shape[0], observation_spec['observation_market'].shape[1], 1)),
            'observation_holdingRate': kr.layers.Dense(1, activation='tanh')
        },
        preprocessing_combiner=kr.layers.Concatenate(axis=-1),
        fc_layer_params=actor_denseLayerParams,
        # enable_last_layer_zero_initializer=False,
        name='ActorNetwork'
    )
    print('Actor Network Created.')
    # Critic Network: we need a Q network to produce f(state, action) -> expected reward(single value)
    # book p.513-515
    # https://www.tensorflow.org/agents/api_docs/python/tf_agents/networks/q_network/QNetwork
    # https://www.tensorflow.org/agents/api_docs/python/tf_agents/networks/value_network/ValueNetwork
    # critic_net = ValueNetwork(
    #     (observation_spec, action_spec),
    #     preprocessing_layers=(
    #         {
    #             'observation_market': kr.models.Sequential([
    #                 kr.layers.Conv2D(filters=int((observation_spec['observation_market'].shape[0]*observation_spec['observation_market'].shape[1])//100), kernel_size=3, activation='relu', input_shape=(observation_spec['observation_market'].shape[0], observation_spec['observation_market'].shape[1], 1)),
    #                 # kr.layers.Conv2D(filters=int((observation_spec[0].shape[0]*observation_spec[0].shape[1])//8), kernel_size=3, activation='relu', input_shape=(observation_spec[0].shape[0], observation_spec[0].shape[1], 1)),
    #                 kr.layers.Flatten()
    #             ]),
    #             'observation_holdingRate': kr.layers.Dense(1, activation='tanh')
    #         },
    #         kr.layers.Dense(1, activation='tanh')
    #     ),
    #     preprocessing_combiner=kr.layers.Concatenate(axis=-1),
    #     conv_layer_params=None,
    #     fc_layer_params=critic_commonDenseLayerParams,
    #     dtype=tf.float32,
    #     name='Critic Network'
    # )
    critic_net = MultiObservationCriticNetwork(
        (observation_spec, action_spec),
        preprocessing_layers={
            'observation_market': kr.models.Sequential([
                kr.layers.Conv2D(filters=int((observation_spec['observation_market'].shape[0]*observation_spec['observation_market'].shape[1])//100), kernel_size=3, activation='relu', input_shape=(observation_spec['observation_market'].shape[0], observation_spec['observation_market'].shape[1], 1)),
                # kr.layers.Conv2D(filters=int((observation_spec[0].shape[0]*observation_spec[0].shape[1])//8), kernel_size=3, activation='relu', input_shape=(observation_spec[0].shape[0], observation_spec[0].shape[1], 1)),
                kr.layers.Flatten()
            ]),
            # 'observation_market': kr.layers.Conv2D(filters=int((observation_spec['observation_market'].shape[0]*observation_spec['observation_market'].shape[1])//100), kernel_size=3, activation='relu', input_shape=(observation_spec['observation_market'].shape[0], observation_spec['observation_market'].shape[1], 1)),
            'observation_holdingRate': kr.layers.Dense(1, activation='tanh')
        },
        preprocessing_combiner=kr.layers.Concatenate(axis=-1),
        joint_fc_layer_params=critic_commonDenseLayerParams,
        joint_activation_fn=tf.nn.relu,
        output_activation_fn=None,
        kernel_initializer=None,
        last_kernel_initializer=None,
        name='Critic Network'
    )
    print('Critic Network Created.')

    # DDPG Agent
    global_step = tf.compat.v1.train.get_or_create_global_step()
    if shouldContinueFromLastCheckpoint:
        global_step = tf.compat.v1.train.get_global_step()
    tf_agent = DdpgAgent(
        time_step_spec=env.time_step_spec(),
        action_spec=action_spec,
        actor_network=actor_net,
        critic_network=critic_net,
        actor_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=actorLearningRate),
        critic_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=criticLearningRate),
        target_update_tau=1.0,
        target_update_period=1,
        dqda_clipping=None,
        td_errors_loss_fn=None,
        gamma=gamma,
        reward_scale_factor=1.0,
        gradient_clipping=None,
        summarize_grads_and_vars=False,
        train_step_counter=global_step,
        name='DDPGAgent'
    )
    tf_agent.initialize()

    # policies
    evaluate_policy = greedy_policy.GreedyPolicy(tf_agent.policy)
    collect_policy = tf_agent.collect_policy

    # metrics and evaluation
    def compute_avg_return(environment, policy, num_episodes=2):
        total_return = 0.0
        for _ in range(num_episodes):
            time_step = environment.reset()
            episode_return = 0.0
            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = environment.step(action_step.action)
                episode_return += time_step.reward
            total_return += episode_return
        avg_return = total_return / num_episodes
        return avg_return.numpy()[0]

    # create Reply Buffer
    # https://www.tensorflow.org/agents/tutorials/5_replay_buffers_tutorial
    # https://www.google.com/url?client=internal-element-cse&cx=016807462989910793636:iigazrvgr1m&q=https://www.tensorflow.org/agents/api_docs/python/tf_agents/replay_buffers/tf_uniform_replay_buffer/TFUniformReplayBuffer&sa=U&ved=2ahUKEwivq9qvnaTrAhXMdXAKHf2nBQYQFjAAegQIBBAB&usg=AOvVaw2elqEhFKUSZf8WeAl53gVK
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=tf_agent.collect_data_spec,
        batch_size=batchSize,
        max_length=replayBufferCapacity
    )
    print(tf_agent.collect_data_spec)
    print('Replay Buffer Created, start warming-up ...')
    _startTime = dt.datetime.now()

    # driver for warm-up
    # https://www.tensorflow.org/agents/api_docs/python/tf_agents/drivers/dynamic_episode_driver/DynamicEpisodeDriver
    initial_collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
        env,
        collect_policy,
        observers=[replay_buffer.add_batch],
        num_episodes=warmupEpisodes
    )
    # run restore process
    if shouldContinueFromLastCheckpoint:
        train_checkpointer = common.Checkpointer(
            ckpt_dir=checkpointDir,
            max_to_keep=1,
            agent=tf_agent,
            policy=tf_agent.policy,
            replay_buffer=replay_buffer,
            global_step=global_step
        )
        train_checkpointer.initialize_or_restore()
    else:
        initial_collect_driver.run()
    _timeCost = (dt.datetime.now() - _startTime).total_seconds()
    print('Replay Buffer Warm-up Done. (cost {:.3g} hours)'.format(_timeCost/3600.0))
    _startTime = dt.datetime.now()


    # Training

    print('Prepare for training ...')
    collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
        env,
        collect_policy,
        observers=[replay_buffer.add_batch],
        num_episodes=collect_episodes_per_iteration
    )
    # # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    # tf_agent.train = common.function(tf_agent.train)
    # collect_driver.run = common.function(collect_driver.run)
    # Reset the train step
    tf_agent.train_step_counter.assign(0)
    # Evaluate the agent's policy once before training.
    # avg_return = compute_avg_return(evaluate_env, evaluate_policy, validateEpisodes)
    # returns = [avg_return]
    # Main training process
    dataset = replay_buffer.as_dataset(num_parallel_calls=7, sample_batch_size=batchSize, num_steps=2)
    iterator = iter(dataset)
    _timeCost = (dt.datetime.now() - _startTime).total_seconds()
    returns = nu.array([])
    steps = nu.array([])
    losses = nu.array([])
    print('All preparation is done (cost {:.3g} hours). Start training...'.format(_timeCost/3600.0))
    _startTimeFromStart = dt.datetime.now()
    for _ in range(num_iterations):
        _startTime = dt.datetime.now()
        # Collect a few steps using collect_policy and save to the replay buffer.
        collect_driver.run()
        # Sample a batch of data from the buffer and update the agent's network.
        experience, unused_info = next(iterator)
        train_loss = tf_agent.train(experience)
        step = tf_agent.train_step_counter.numpy()
        # show the loss and time cost
        if step % log_interval == 0:
            _timeCost = (dt.datetime.now() - _startTime).total_seconds()
            _timeCostFromStart = (dt.datetime.now() - _startTimeFromStart).total_seconds()
            if _timeCost <= 60:
                print('step = {:>5}: loss = {:+10.6f}  (cost {:>5.2f} [sec]; {:>.2f} [hrs] from start.)'.format(step, train_loss.loss, _timeCost, _timeCostFromStart/3600.0))
            elif _timeCost <= 3600:
                print('step = {:>5}: loss = {:+10.6f}  (cost {:>5.2f} [min]; {:>.2f} [hrs] from start.)'.format(step, train_loss.loss, _timeCost/60.0, _timeCostFromStart/3600.0))
            else:
                print('step = {:>5}: loss = {:+10.6f}  (cost {:>5.2f} [hrs]; {:>.2f} [hrs] from start.)'.format(step, train_loss.loss, _timeCost/3600.0, _timeCostFromStart/3600.0))
        if step % eval_interval == 0:
            avg_return = compute_avg_return(evaluate_env, evaluate_policy, validateEpisodes)
            print('step = {0}: Average Return = {1}'.format(step, avg_return))
            steps = nu.append(steps, step)
            returns = nu.append(returns, avg_return)
            losses = nu.append(losses, train_loss.loss)
            # save temp results
            with open('DDPGAgent_tempResults.pickle', 'wb') as file:
                pickle.dump(nu.concatenate([steps.reshape(-1, 1), returns.reshape(-1, 1), losses.reshape(-1, 1)], axis=-1), file)
            # save models
            # a checkpoint of a agent model can be used to restart a training
            # https://www.tensorflow.org/agents/tutorials/10_checkpointer_policysaver_tutorial?hl=en
            train_checkpointer = common.Checkpointer(
                ckpt_dir=checkpointDir,
                max_to_keep=1,
                agent=tf_agent,
                policy=tf_agent.policy,
                replay_buffer=replay_buffer,
                global_step=global_step
            )
            train_checkpointer.save(global_step)
            # # save policy
            # # saved policies can only be used to evaluate, not to train.
            # tf_policy_saver = policy_saver.PolicySaver(evaluate_policy)
            # tf_policy_saver.save(policy_dir)
    # save results
    with open('DDPGAgent_results.pickle', 'wb') as file:
        pickle.dump(nu.concatenate([steps.reshape(-1, 1), returns.reshape(-1, 1), losses.reshape(-1, 1)], axis=-1), file)
    # save models
    # a checkpoint of a agent model can be used to restart a training
    # https://www.tensorflow.org/agents/tutorials/10_checkpointer_policysaver_tutorial?hl=en
    train_checkpointer = common.Checkpointer(
        ckpt_dir=checkpointDir,
        max_to_keep=1,
        agent=tf_agent,
        policy=tf_agent.policy,
        replay_buffer=replay_buffer,
        global_step=global_step
    )
    train_checkpointer.save(global_step)
    # # save policy
    # # saved policies can only be used to evaluate, not to train.
    # tf_policy_saver = policy_saver.PolicySaver(evaluate_policy)
    # tf_policy_saver.save(policy_dir)
    # plot
    pl.xlabel('Step', fontsize=22)
    pl.ylabel('Returns', fontsize=22)
    pl.tick_params(labelsize=16)
    pl.plot(steps, returns)
    pl.show()

    pl.xlabel('Step', fontsize=22)
    pl.ylabel('Loss', fontsize=22)
    pl.tick_params(labelsize=16)
    pl.plot(steps, losses)
    pl.show()
