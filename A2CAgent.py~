from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64
import numpy as nu
import pandas as pd
import datetime as dt
import multiprocessing as mp
from matplotlib import pyplot as pl
import pickle
import tensorflow as tf
tf.compat.v1.enable_v2_behavior()
from tensorflow import keras as kr
from tf_agents.agents.reinforce.reinforce_agent import ReinforceAgent
from tf_agents.networks import encoding_network, utils
from tf_agents.networks import actor_distribution_network
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.networks.network import Network
from tf_agents.networks.value_network import ValueNetwork
from tf_agents.drivers import dynamic_step_driver, dynamic_episode_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.experimental.train.utils import spec_utils, strategy_utils, train_utils
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.policies import greedy_policy, random_tf_policy
from tf_agents.utils import common, nest_utils
# Custom Modules
from BitcoinEnvironment import BTC_JPY_Environment



if __name__ == '__main__':
    mp.freeze_support()
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
    batchSize = 1
    num_iterations = int(1e5)
    collect_episodes_per_iteration = 10
    _storeFullEpisodes = collect_episodes_per_iteration
    replayBufferCapacity = int(_storeFullEpisodes * episodeEndSteps * batchSize)

    # observationConvParams = [(int(observation_spec['observation_market'].shape[0]//100), 3, 1)]
    critic_commonDenseLayerParams = [int(observation_spec['observation_market'].shape[0]//100)]
    actor_denseLayerParams = [int(observation_spec['observation_market'].shape[0]//100)]

    learning_rate = 1e-6 # @param {type:"number"}
    entropy_coeff = 0.1
    log_interval = 25 # @param {type:"integer"}
    eval_interval = 1000 # @param {type:"integer"}
    validateEpisodes = 3


    # Actor Network
    # A2C or REINFORCE Agent has an actor giving the distribution (or the mean and variance) of actions,
    # so an actorDistributionNetwork is needed, not ones who directly return actions.
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        input_tensor_spec=observation_spec,
        output_tensor_spec=action_spec,
        preprocessing_layers={
            'observation_market': kr.models.Sequential([
                kr.layers.Conv2D(filters=int((observation_spec['observation_market'].shape[0]*observation_spec['observation_market'].shape[1])//100), kernel_size=3, activation='relu', input_shape=(observation_spec['observation_market'].shape[0], observation_spec['observation_market'].shape[1], 1)),
                kr.layers.Flatten()
            ]),
            # 'observation_market': kr.layers.Conv2D(filters=int((observation_spec['observation_market'].shape[0]*observation_spec['observation_market'].shape[1])//100), kernel_size=3, activation='relu', input_shape=(observation_spec['observation_market'].shape[0], observation_spec['observation_market'].shape[1], 1)),
            'observation_holdingRate': kr.layers.Dense(1, activation='tanh')
        },
        preprocessing_combiner=kr.layers.Concatenate(axis=-1),
        fc_layer_params=actor_denseLayerParams,
        dtype=tf.float32,
        continuous_projection_net=tanh_normal_projection_network.TanhNormalProjectionNetwork,
        name='ActorDistributionNetwork'
    )
    print('Actor Network Created.')

    # Critic Network
    critic_net = ValueNetwork(
        observation_spec,
        preprocessing_layers={
            'observation_market': kr.models.Sequential([
                kr.layers.Conv2D(filters=int((observation_spec['observation_market'].shape[0]*observation_spec['observation_market'].shape[1])//100), kernel_size=3, activation='relu', input_shape=(observation_spec['observation_market'].shape[0], observation_spec['observation_market'].shape[1], 1)),
                kr.layers.Flatten()
            ]),
            # 'observation_market': kr.layers.Conv2D(filters=int((observation_spec['observation_market'].shape[0]*observation_spec['observation_market'].shape[1])//100), kernel_size=3, activation='relu', input_shape=(observation_spec['observation_market'].shape[0], observation_spec['observation_market'].shape[1], 1)),
            'observation_holdingRate': kr.layers.Dense(1, activation='tanh')
        },
        preprocessing_combiner=kr.layers.Concatenate(axis=-1),
        conv_layer_params=None,
        fc_layer_params=critic_commonDenseLayerParams,
        dtype=tf.float32,
        name='Critic Network'
    )

    # Agent
    # https://www.tensorflow.org/agents/api_docs/python/tf_agents/agents/ReinforceAgent
    global_step = tf.compat.v1.train.get_or_create_global_step()
    tf_agent = ReinforceAgent(
        time_step_spec=env.time_step_spec(),
        action_spec=action_spec,
        actor_network=actor_net,
        optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate),
        value_network=critic_net,
        value_estimation_loss_coef=0.2,
        advantage_fn=lambda returns, value_preds: returns - value_preds,
        use_advantage_loss=True,
        gamma=gamma,
        normalize_returns=True,
        debug_summaries=True,
        summarize_grads_and_vars=True,
        entropy_regularization=None,
        train_step_counter=global_step,
        name='A2CAgent'
    )
    tf_agent.initialize()
    print('A2C Agent Created.')

    # Policies
    evaluate_policy = tf_agent.policy
    collect_policy = tf_agent.collect_policy

    # Evaluation
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

    # Replay Buffer
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=tf_agent.collect_data_spec,
        batch_size=batchSize,
        max_length=replayBufferCapacity
    )
    print('Replay Buffer Created, start warming-up ...')
    _startTime = dt.datetime.now()

    # Drivers
    # initial_collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
    #     env,
    #     collect_policy,
    #     observers=[replay_buffer.add_batch],
    #     num_episodes=_storeFullEpisodes
    # )
    # initial_collect_driver.run()
    # _timeCost = (dt.datetime.now() - _startTime).total_seconds()
    # print('Replay Buffer Warm-up Done. (cost {:.3g} hours)'.format(_timeCost/3600.0))
    _startTime = dt.datetime.now()

    print('Prepare for training ...')
    collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
        env,
        collect_policy,
        observers=[replay_buffer.add_batch],
        num_episodes=2
    )
    tf_agent.train_step_counter.assign(0)

    # Initialize avg_return
    # avg_return = compute_avg_return(evaluate_env, evaluate_policy, 1)
    # returns = [avg_return]

    # Training
    returns = []
    steps = []
    losses = []
    _timeCost = (dt.datetime.now() - _startTime).total_seconds()
    print('All preparation is done (cost {:.3g} hours). Start training...'.format(_timeCost/3600.0))
    _startTimeFromStart = dt.datetime.now()
    for _ in range(num_iterations):
        _startTime = dt.datetime.now()
        # Collect a few steps using collect_policy and save to the replay buffer.
        collect_driver.run()
        # Since the A2C is a on-policy method, we gather all experiences for one descent step and have the buffer cleared immediately.
        experience = replay_buffer.gather_all()
        train_loss = tf_agent.train(experience)
        replay_buffer.clear()
        step = tf_agent.train_step_counter.numpy()
        # if step % log_interval == 0:
        # print time cost and loss
        _timeCost = (dt.datetime.now() - _startTime).total_seconds()
        _timeCostFromStart = (dt.datetime.now() - _startTimeFromStart).total_seconds()
        if _timeCost <= 60:
            print('step = {:>5}: loss = {:+10.6f}  (cost {:>5.2f} [sec]; {:>.2f} [hrs] from start.)'.format(step, train_loss.loss, _timeCost, _timeCostFromStart/3600.0))
        elif _timeCost <= 3600:
            print('step = {:>5}: loss = {:+10.6f}  (cost {:>5.2f} [min]; {:>.2f} [hrs] from start.)'.format(step, train_loss.loss, _timeCost/60.0, _timeCostFromStart/3600.0))
        else:
            print('step = {:>5}: loss = {:+10.6f}  (cost {:>5.2f} [hrs]; {:>.2f} [hrs] from start.)'.format(step, train_loss.loss, _timeCost/3600.0, _timeCostFromStart/3600.0))
        # evaluate policy and show average return
        if step % eval_interval == 0:
            avg_return = compute_avg_return(evaluate_env, evaluate_policy, validateEpisodes)
            print('step = {:>5}: Average Return = {}'.format(step, avg_return))
            returns.append(avg_return)
            steps.append(step)
            losses.append(train_loss.loss)
    # change format
    returns = nu.array(returns)
    steps = nu.array(steps)
    losses = nu.array(losses)
    # save results
    with open('A2CAgent_results.pickle', 'wb') as file:
        pickle.dump(nu.concatenate([steps, returns, losses]), file)
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
