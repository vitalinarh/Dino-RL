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

    def predict(self, inputs):
        return self.model(np.atleast_2d(inputs.astype('float32')))

    def train(self, TargetNet):
        if len(self.experience['s']) < self.min_experiences:
            return 0
        ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size)
        states = np.asarray([self.experience['s'][i] for i in ids])
        actions = np.asarray([self.experience['a'][i] for i in ids])
        rewards = np.asarray([self.experience['r'][i] for i in ids])
        states_next = np.asarray([self.experience['s2'][i] for i in ids])
        dones = np.asarray([self.experience['done'][i] for i in ids])
        value_next = np.max(TargetNet.predict(states_next), axis=1)
        actual_values = np.where(dones, rewards, rewards+self.gamma*value_next)

        with tf.GradientTape() as tape:
            selected_action_values = tf.math.reduce_sum(
                self.predict(states) * tf.one_hot(actions, self.num_actions), axis=1)
            loss = tf.math.reduce_mean(tf.square(actual_values - selected_action_values))
        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss

    def get_action(self, states, epsilon):
        if np.random.random() < epsilon:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.predict(np.atleast_2d(states))[0])

    def add_experience(self, exp):
        if len(self.experience['s']) >= self.max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0)
        for key, value in exp.items():
            self.experience[key].append(value)

    def copy_weights(self, TrainNet):
        variables1 = self.model.trainable_variables
        variables2 = TrainNet.model.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())

def play(env, TrainNet, TargetNet, epsilon, decay, copy_step):

    rewards = 0
    done = False
    observations = env.reset()
    losses = []
    i = 0

    while True:
        epsilon = max(min_epsilon, epsilon * decay)
        prev_observations = observations
        if done:
            score = env.unwrapped.game.get_score()
            rewards = score
            print("iteration: ", i, "total reward: ", rewards)
            env.reset()
        action = TrainNet.get_action(observations, epsilon)
        observation, reward, done, info = env.step(action)
        rewards += reward

        exp = {'s': prev_observations, 'a': action, 'r': reward, 's2': observations, 'done': done}
        TrainNet.add_experience(exp)
        loss = TrainNet.train(TargetNet)
        if isinstance(loss, int):
            losses.append(loss)
        else:
            losses.append(loss.numpy())
        i += 1
        if i % copy_step == 0:
            TargetNet.copy_weights(TrainNet)

    return rewards, np.mean(losses)

if __name__ == "__main__":
    env = gym.make('ChromeDino-v0')
    env.reset()

    num_states = len(env.observation_space.sample())
    num_actions = env.action_space.n

    gamma = 0.99 #  decay rate of past observations original 0.99
    copy_step = 25

    hidden_units = [200, 200]
    max_experiences = 10000
    min_experiences = 100 # # timesteps to observe before training

    batch_size = 1 # # size of minibatch, keep it low
    lr = 1e-2

    TrainNet = DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr)
    TargetNet = DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr)

    N = 1000
    epsilon = 0.99 # # starting value of epsilon
    decay = 0.9999
    min_epsilon = 0.1 # # final value of epsilon

    play(env, TrainNet, TargetNet, epsilon, decay, copy_step)
