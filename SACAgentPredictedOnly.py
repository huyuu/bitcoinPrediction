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
from tf_agents.agents.ddpg import critic_network
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
from tf_agents.utils import common
import shutil
# Custom Modules
from BitcoinEnvironmentPredictedOnly import BTC_JPY_Environment
from MultiObservationCriticNetwork import MultiObservationCriticNetwork


def create_zip_file(dirname, base_filename):
  return shutil.make_archive(base_filename, 'zip', dirname)

def upload_and_unzip_file_to(dirname):
    if files is None:
        return
    uploaded = files.upload()
    for fn in uploaded.keys():
        print('User uploaded file "{name}" with length {length} bytes'.format(
            name=fn, length=len(uploaded[fn])))
    shutil.rmtree(dirname)
    zip_files = zipfile.ZipFile(io.BytesIO(uploaded[fn]), 'r')
    zip_files.extractall(dirname)
    zip_files.close()


if __name__ == '__main__':
    mp.freeze_support()
    # check if should continue from last stored checkpoint
    if len(sys.argv) == 2:
        shouldContinueFromLastCheckpoint = sys.argv[1] == '-c'
    else:
        shouldContinueFromLastCheckpoint = False

    # Environment

    # create environment and transfer it to Tensorflow version
    gamma = 0.999
    print('Creating environment ...')
    env = BTC_JPY_Environment(imageWidth=int(24*4), imageHeight=int(24*8), initialAsset=100000, isHugeMemorryMode=True, shouldGiveRewardsFinally=True, gamma=gamma)
    episodeEndSteps = env.episodeEndSteps
    env = tf_py_environment.TFPyEnvironment(env)
    evaluate_env = tf_py_environment.TFPyEnvironment(BTC_JPY_Environment(imageWidth=int(24*4), imageHeight=int(24*8), initialAsset=100000, isHugeMemorryMode=False, shouldGiveRewardsFinally=True, gamma=gamma))
    observation_spec = env.observation_spec()
    action_spec = env.action_spec()
    print('Environment created.')

    # # create strategy
    # strategy = strategy_utils.get_strategy(tpu=False, use_gpu=False)


    # Hyperparameters

    batchSize = 1
    num_iterations = int(1e5)
    log_interval = num_iterations//1000
    eval_interval = num_iterations//100

    criticLearningRate = 1e-4
    actorLearningRate = 1e-4
    alphaLearningRate = 1e-4

    gradientClipping = None
    target_update_tau = 1e-4

    # (num_units, kernel_size, stride)
    # critic_observationConvLayerParams = [int(observation_spec['observation_market'].shape[0]//4)]
    # critic_commonDenseLayerParams = [int(observation_spec['observation_market'].shape[0]//50), int(observation_spec['observation_market'].shape[0]//50)]
    # actor_denseLayerParams = [int(observation_spec['observation_market'].shape[0]//50), int(observation_spec['observation_market'].shape[0]//50)]

    collect_episodes_per_iteration = 10
    _storeFullEpisodes = 200
    replayBufferCapacity = int(_storeFullEpisodes * episodeEndSteps * batchSize)
    warmupEpisodes = _storeFullEpisodes
    validateEpisodes = 10

    checkpointDir = './SACAgent_checkcpoints'
    if not os.path.exists(checkpointDir):
        os.mkdir(checkpointDir)

    policyDir = './SACAgent_savedPolicy'
    if not os.path.exists(policyDir):
        os.mkdir(policyDir)

    # Models

    # create Crite Network
    # https://www.tensorflow.org/agents/api_docs/python/tf_agents/agents/ddpg/critic_network/CriticNetwork
    # critic_net = value_network.ValueNetwork(
    #     (observation_spec, action_spec),
    #     preprocessing_layers=(
    #         {
    #             'observation_market': kr.models.Sequential([
    #                 kr.layers.Conv2D(filters=int((observation_spec['observation_market'].shape[0]*observation_spec['observation_market'].shape[1])//8), kernel_size=3, activation='relu', input_shape=(observation_spec['observation_market'].shape[0], observation_spec['observation_market'].shape[1], 1)),
    #                 # kr.layers.Conv2D(filters=int((observation_spec[0].shape[0]*observation_spec[0].shape[1])//8), kernel_size=3, activation='relu', input_shape=(observation_spec[0].shape[0], observation_spec[0].shape[1], 1)),
    #                 kr.layers.Flatten()
    #             ]),
    #             'observation_holdingRate': kr.layers.Dense(1, activation='sigmoid')
    #         },
    #         kr.layers.Dense(1, activation='sigmoid')
    #     ),
    #     preprocessing_combiner=kr.layers.Concatenate(axis=-1),
    #     conv_layer_params=None,
    #     fc_layer_params=critic_commonDenseLayerParams,
    #     dtype=tf.float32,
    #     name='Critic Network'
    # )
    # critic_net = MultiObservationCriticNetwork(
    #     (observation_spec, action_spec),
    #     preprocessing_layers={
    #         'observation_market': kr.models.Sequential([
    #             kr.layers.Conv2D(filters=int((observation_spec['observation_market'].shape[0]*observation_spec['observation_market'].shape[1])//100), kernel_size=3, activation='relu', input_shape=(observation_spec['observation_market'].shape[0], observation_spec['observation_market'].shape[1], 1)),
    #             kr.layers.Conv2D(filters=int((observation_spec['observation_market'].shape[0]*observation_spec['observation_market'].shape[1])//100), kernel_size=3, activation='relu', input_shape=(observation_spec['observation_market'].shape[0], observation_spec['observation_market'].shape[1], 1)),
    #             kr.layers.Flatten(),
    #             kr.layers.Dense(5, activation='tanh'),
    #             kr.layers.Flatten()
    #         ]),
    #         # 'observation_market': kr.layers.Conv2D(filters=int((observation_spec['observation_market'].shape[0]*observation_spec['observation_market'].shape[1])//100), kernel_size=3, activation='relu', input_shape=(observation_spec['observation_market'].shape[0], observation_spec['observation_market'].shape[1], 1)),
    #         'observation_holdingRate': kr.layers.Dense(2, activation='tanh')
    #     },
    #     preprocessing_combiner=kr.layers.Concatenate(axis=-1),
    #     joint_fc_layer_params=critic_commonDenseLayerParams,
    #     joint_activation_fn=tf.nn.relu,
    #     output_activation_fn=None,
    #     kernel_initializer=None,
    #     last_kernel_initializer=None,
    #     name='Critic Network'
    # )
    critic_net = critic_network.CriticNetwork(
        (observation_spec, action_spec),
        observation_fc_layer_params=[4, 4],
        action_fc_layer_params=[2],
        joint_fc_layer_params=[2, 2],
        kernel_initializer='glorot_uniform',
        last_kernel_initializer='glorot_uniform'
    )
    print('Critic Network Created.')

    # # create Actor Network
    # def normal_projection_net(action_spec):
    #     return normal_projection_network.NormalProjectionNetwork(
    #         action_spec,
    #         mean_transform=None,
    #         state_dependent_std=True,
    #         init_means_output_factor=0.1,
    #         std_transform=sac_agent.std_clip_transform,
    #         scale_distribution=True
    #     )
    # https://www.tensorflow.org/agents/api_docs/python/tf_agents/networks/actor_distribution_network/ActorDistributionNetwork
    # with strategy.scope():
    # actor_net = actor_distribution_network.ActorDistributionNetwork(
    #     input_tensor_spec=observation_spec,
    #     output_tensor_spec=action_spec,
    #     preprocessing_layers={
    #         'observation_market': kr.models.Sequential([
    #             kr.layers.Conv2D(filters=int((observation_spec['observation_market'].shape[0]*observation_spec['observation_market'].shape[1])//100), kernel_size=3, activation='relu', input_shape=(observation_spec['observation_market'].shape[0], observation_spec['observation_market'].shape[1], 1)),
    #             kr.layers.Conv2D(filters=int((observation_spec['observation_market'].shape[0]*observation_spec['observation_market'].shape[1])//100), kernel_size=3, activation='relu', input_shape=(observation_spec['observation_market'].shape[0], observation_spec['observation_market'].shape[1], 1)),
    #             kr.layers.Flatten(),
    #             kr.layers.Dense(5, activation='tanh'),
    #             kr.layers.Flatten()
    #         ]),
    #         'observation_holdingRate': kr.layers.Dense(2, activation='tanh')
    #     },
    #     preprocessing_combiner=kr.layers.Concatenate(axis=-1),
    #     fc_layer_params=actor_denseLayerParams,
    #     dtype=tf.float32,
    #     continuous_projection_net=tanh_normal_projection_network.TanhNormalProjectionNetwork,
    #     name='ActorDistributionNetwork'
    # )
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        observation_spec,
        action_spec,
        fc_layer_params=[4, 4],
        continuous_projection_net=(
        tanh_normal_projection_network.TanhNormalProjectionNetwork)
    )
    print('Actor Network Created.')

    # create SAC Agent
    # https://www.tensorflow.org/agents/api_docs/python/tf_agents/agents/SacAgent
    global_step = tf.compat.v1.train.get_or_create_global_step()
    if shouldContinueFromLastCheckpoint:
        global_step = tf.compat.v1.train.get_global_step()
    # with strategy.scope():
    #     train_step = train_utils.create_train_step()
    tf_agent = sac_agent.SacAgent(
        env.time_step_spec(),
        action_spec,
        actor_network=actor_net,
        critic_network=critic_net,
        actor_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=actorLearningRate),
        critic_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=criticLearningRate),
        alpha_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=alphaLearningRate),
        target_update_tau=target_update_tau,
        gamma=gamma,
        gradient_clipping=gradientClipping,
        train_step_counter=global_step,
    )
    tf_agent.initialize()
    print('SAC Agent Created.')


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
    print('All preparation is done (cost {:.3g} hours). Start training...'.format(_timeCost/3600.0))
    returns = nu.array([])
    steps = nu.array([])
    losses = nu.array([])
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
            with open('SACAgent_tempResults.pickle', 'wb') as file:
                pickle.dump(nu.concatenate([steps.reshape(-1, 1), returns.reshape(-1, 1), losses.reshape(-1, 1)], axis=-1), file)
            # # save models
            # # a checkpoint of a agent model can be used to restart a training
            # # https://www.tensorflow.org/agents/tutorials/10_checkpointer_policysaver_tutorial?hl=en
            # train_checkpointer = common.Checkpointer(
            #     ckpt_dir=checkpointDir,
            #     max_to_keep=1,
            #     agent=tf_agent,
            #     policy=tf_agent.policy,
            #     replay_buffer=replay_buffer,
            #     global_step=global_step
            # )
            # train_checkpointer.save(global_step)
            # # save policy
            # # saved policies can only be used to evaluate, not to train.
            # tf_policy_saver = policy_saver.PolicySaver(evaluate_policy)
            # tf_policy_saver.save(policy_dir)
    # save results
    with open('SACAgent_results.pickle', 'wb') as file:
        pickle.dump(nu.concatenate([steps.reshape(-1, 1), returns.reshape(-1, 1), losses.reshape(-1, 1)], axis=-1), file)
    # save models
    # a checkpoint of a agent model can be used to restart a training
    # https://www.tensorflow.org/agents/tutorials/10_checkpointer_policysaver_tutorial?hl=en
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
    # save
    train_checkpointer = common.Checkpointer(
        ckpt_dir=checkpointDir,
        max_to_keep=1,
        agent=tf_agent,
        policy=tf_agent.policy,
        replay_buffer=replay_buffer,
        global_step=global_step
    )
    train_checkpointer.save(global_step)
