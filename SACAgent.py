# Importing
# Python Modules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as nu
import pandas as pd
import datetime as dt
# Tensorflow Modules
import tensorflow as tf
tf.compat.v1.enable_v2_behavior()
from tensorflow import keras as kr
from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent
from tf_agents.drivers import dynamic_step_driver, dynamic_episode_driver
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_network, normal_projection_network, value_network
from tf_agents.policies import greedy_policy, random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
# Custom Modules
from BitcoinEnvironment import BTC_JPY_Environment



# Environment

# create environment and transfer it to Tensorflow version
env = BTC_JPY_Environment(imageWidth=int(24*4), imageHeight=int(24*8), initialAsset=100000)
env = tf_py_environment.TFPyEnvironment(env)
eval_env = tf_py_environment.TFPyEnvironment(BTC_JPY_Environment(imageWidth=int(24*4), imageHeight=int(24*8), initialAsset=100000))
observation_spec = env.observation_spec()
action_spec = env.action_spec()


# Hyperparameters

batchSize = int(4*24/2)

criticLearningRate = 3e-4
actorLearningRate = 3e-4
alphaLearningRate = 3e-4

gamma = 0.999
gradientClipping = None
target_update_tau = 0.005

# (num_units, kernel_size, stride)
# critic_observationConvLayerParams = [(24, 3, 1), (24, 3, 1)]
# critic_observationDenseLayerParams = [int(env.observation_spec()[0].shape[0]//2), int(env.observation_spec()[0].shape[0]//2)]
critic_commonDenseLayerParams = [int(observation_spec[0].shape[0]//2), int(observation_spec[0].shape[0]//2)]
# actor_convLayerParams = [(96, 3, 1), (24, 3, 1)]
actor_denseLayerParams = [int(observation_spec[0].shape[0]//2), int(observation_spec[0].shape[0]//2)]

replayBufferCapacity = 100

warmupEpisodes = 80



# Models

# create Crite Network
# https://www.tensorflow.org/agents/api_docs/python/tf_agents/agents/ddpg/critic_network/CriticNetwork
# critic_net = critic_network.CriticNetwork(
#     (observation_spec, action_spec),
#     observation_fc_layer_params=critic_observationDenseLayerParams,
#     action_fc_layer_params=None,
#     joint_fc_layer_params=critic_commonDenseLayerParams
# )
critic_net = value_network.ValueNetwork(
    (observation_spec, action_spec),
    preprocessing_layers=(
        (
            kr.models.Sequential([
                kr.layers.Conv2D(filters=int((observation_spec[0].shape[0]*observation_spec[0].shape[1])//3), kernel_size=3, activation='relu', input_shape=(observation_spec[0].shape[0], observation_spec[0].shape[1], 1)),
                kr.layers.Conv2D(filters=int((observation_spec[0].shape[0]*observation_spec[0].shape[1])//3), kernel_size=3, activation='relu', input_shape=(observation_spec[0].shape[0], observation_spec[0].shape[1], 1)),
                kr.layers.Flatten()
            ]),
            kr.layers.Dense(1, activation='sigmoid')
        ),
        kr.layers.Dense(1, activation='sigmoid')
    ),
    preprocessing_combiner=kr.layers.Concatenate(axis=-1),
    conv_layer_params=None,
    fc_layer_params=critic_commonDenseLayerParams,
    dtype=tf.float32,
    name='Critic Network'
)
print('Critic Network Created.')

# create Actor Network
def normal_projection_net(action_spec,init_means_output_factor=0.1):
    return normal_projection_network.NormalProjectionNetwork(
        action_spec,
        mean_transform=None,
        state_dependent_std=True,
        init_means_output_factor=init_means_output_factor,
        std_transform=sac_agent.std_clip_transform,
        scale_distribution=True
    )
# https://www.tensorflow.org/agents/api_docs/python/tf_agents/networks/actor_distribution_network/ActorDistributionNetwork
actor_net = actor_distribution_network.ActorDistributionNetwork(
    input_tensor_spec=observation_spec,
    output_tensor_spec=action_spec,
    preprocessing_layers=(
        kr.models.Sequential([
            kr.layers.Conv2D(filters=int((observation_spec[0].shape[0]*observation_spec[0].shape[1])//3), kernel_size=3, activation='relu', input_shape=(observation_spec[0].shape[0], observation_spec[0].shape[1], 1)),
            kr.layers.Conv2D(filters=int((observation_spec[0].shape[0]*observation_spec[0].shape[1])//3), kernel_size=3, activation='relu', input_shape=(observation_spec[0].shape[0], observation_spec[0].shape[1], 1)),
            kr.layers.Flatten()
        ]),
        kr.layers.Dense(1, activation='sigmoid')
    ),
    preprocessing_combiner=kr.layers.Concatenate(axis=-1),
    fc_layer_params=actor_denseLayerParams,
    dtype=tf.float32,
    continuous_projection_net=normal_projection_net,
    name='ActorDistributionNetwork'
)
print('Actor Network Created.')

# create SAC Agent
# https://www.tensorflow.org/agents/api_docs/python/tf_agents/agents/SacAgent
global_step = tf.compat.v1.train.get_or_create_global_step()
tf_agent = sac_agent.SacAgent(
    env.time_step_spec(),
    action_spec,
    actor_network=actor_net,
    critic_network=critic_net,
    actor_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=actorLearningRate),
    critic_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=criticLearningRate),
    alpha_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=alphaLearningRate),
    target_update_tau=target_update_tau,
    td_errors_loss_fn=tf.compat.v1.losses.mean_squared_error,
    gamma=gamma,
    gradient_clipping=gradientClipping,
    train_step_counter=global_step
)
tf_agent.initialize()
print('SAC Agent Created.')

# policies
evaluationPolicy = greedy_policy.GreedyPolicy(tf_agent.policy)
collectingPolicy = tf_agent.collect_policy

# metrics and evaluation
def compute_avg_return(environment, policy, num_episodes=5):
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
    batch_size=env.batch_size,
    max_length=replayBufferCapacity
)
print('Replay Buffer Created')

# driver for warm-up
# https://www.tensorflow.org/agents/api_docs/python/tf_agents/drivers/dynamic_episode_driver/DynamicEpisodeDriver
initial_collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
    env,
    collect_policy,
    observers=[replay_buffer.add_batch],
    num_episodes=warmupEpisodes
)
initial_collect_driver.run()
print('Replay Buffer Warm-up Done.')


# Training

collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
    train_env,
    collect_policy,
    observers=[replay_buffer.add_batch],
    num_episodes=1
)
# (Optional) Optimize by wrapping some of the code in a graph using TF function.
tf_agent.train = common.function(tf_agent.train)
collect_driver.run = common.function(collect_driver.run)
# Reset the train step
tf_agent.train_step_counter.assign(0)
# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(eval_env, eval_policy, num_eval_episodes)
returns = [avg_return]
# Main training process
dataset = replay_buffer.as_dataset(num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2).prefetch(3)
iterator = iter(dataset)
print('Start training...')
for _ in range(num_iterations):
    # Collect a few steps using collect_policy and save to the replay buffer.
    collect_driver.run()
    # Sample a batch of data from the buffer and update the agent's network.
    experience, unused_info = next(iterator)
    train_loss = tf_agent.train(experience)
    step = tf_agent.train_step_counter.numpy()
    if step % log_interval == 0:
        print('step = {0}: loss = {1}'.format(step, train_loss.loss))
    if step % eval_interval == 0:
        avg_return = compute_avg_return(eval_env, eval_policy, num_eval_episodes)
        print('step = {0}: Average Return = {1}'.format(step, avg_return))
    returns.append(avg_return)
