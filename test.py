import gym
import gym_chrome_dino
from gym_chrome_dino.utils.wrappers import make_dino
import numpy as np
import tensorflow as tf
import os
import datetime

class Model(tf.keras.Model):
    # define layers with __init__
    # input shape is [batch size, size of a state (# in this case)]
    # output shape is [batch size, number of actions (2 in this case)].
    def __init__(self, num_states, hidden_units, num_actions):
        super(Model, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(num_states,))
        self.hidden_layers = []
        for i in hidden_units:
            self.hidden_layers.append(tf.keras.layers.Dense(i, activation='tanh', kernel_initializer='RandomNormal'))
        self.output_layer = tf.keras.layers.Dense(num_actions, activation='linear', kernel_initializer='RandomNormal')

    # implement the model's forward pass
    # @tf.function enables autograph and automatic control dependencies
    @tf.function
    def call(self, inputs):
        z = self.input_layer(inputs)
        for layer in self.hidden_layers:
            z = layer(z)
        output = self.output_layer(z)
        return output

# create the Deep Q-Net model
# define the number of actions, batch size and the optimizer for gradient descent
# gamma is a value between 0 and 1 that is multiplied by the Q value at the next step
# we also initialize MyModel as an instance variable self.mode and create the experience replay buffer self.experience
# The agent won’t start learning unless the size the buffer is greater than self.min_experience, and once the buffer
# reaches the max size self.max_experience, it will delete the oldest values to make room for the new values.
class DQN:
    def __init__(self, num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr):
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.optimizer = tf.optimizers.Adam(lr)
        self.gamma = gamma
        self.model = Model(num_states, hidden_units, num_actions)
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences

def play(env, TrainNet, TargetNet, epsilon, copy_step):

    rewards = 0
    done = False
    observations = env.reset()

    while True:
        if done:
            score = env.unwrapped.game.get_score()
            print(score)
            env.reset()
        action = TrainNet.get_action(observations, epsilon)
        observation, reward, done, info = env.step(action)

    return rewards, np.mean(losses)

if __name__ == "__main__":
    env = gym.make('ChromeDino-v0')
    env.reset()

    num_states = len(env.observation_space.sample())
    num_actions = env.action_space.n

    gamma = 0.99
    copy_step = 25

    hidden_units = [200, 200]
    max_experiences = 10000
    min_experiences = 100

    batch_size = 32
    lr = 1e-2

    TrainNet = DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr)
    TargetNet = DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr)

    N = 50000
    epsilon = 0.99
    decay = 0.9999
    min_epsilon = 0.1

    for n in range(N):
        epsilon = max(min_epsilon, epsilon * decay)
        total_reward, losses = play(env, TrainNet, TargetNet, epsilon, copy_step)

        if n % 100 == 0:
            print("episode: ",  n, "episode reward: ", total_reward, "eps: ", epsilon, "episode loss: ", losses)

        env.close()
