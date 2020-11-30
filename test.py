import gym
import gym_chrome_dino
from gym_chrome_dino.utils.wrappers import make_dino
import random
import numpy as np
from collections import deque
import tensorflow as tf
from keras.models import Sequential
from keras.layers import InputLayer, Dense, Activation, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam
import os
import datetime

class DQN_Agent:
    #
    # Initialize attributes and constructs CNN train_model and target_model
    #
    def __init__(self, state_shape, action_size, gamma, epsilon, min_epsilon, decay, lr, update_rate, max_experiences, min_experiences):
        self.state_shape = state_shape
        self.action_size = action_size
        self.min_experiences = min_experiences
        self.max_experiences = max_experiences
        self.memory = deque(maxlen=self.max_experiences)

        # Hyperparameters
        self.gamma = gamma                  # Discount rate
        self.epsilon = epsilon              # Exploration rate
        self.min_epsilon = min_epsilon      # Minimal exploration rate (epsilon-greedy)
        self.decay = decay                  # Decay rate for epsilon
        self.lr = lr                        # LEarning rate
        self.update_rate = update_rate      # Number of steps until updating the target network

        # Construct DQN models
        self.train_model = self._build_model()
        self.target_model = self._build_model()

        self.train_model.summary()

    #
    # Constructs model
    #
    def _build_model(self):
        model = Sequential()

        # model.add(Dense(512, activation='relu', input_shape=self.state_shape))
        # Convolutional layers
        model.add(Conv2D(32, (8, 8), strides=4, padding='same', input_shape=self.state_shape))
        model.add(Activation('relu'))

        # model.add(Conv2D(64, (4, 4), strides=2, padding='same'))
        # model.add(Activation('relu'))

        # model.add(Conv2D(64, (3, 3), strides=1, padding='same'))
        # model.add(Activation('relu'))
        # model.add(Flatten())

        # FC Layers
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))

        model.compile(loss='mse', optimizer=Adam())
        return model

    #
    # Stores experience in replay memory
    #
    def add_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    #
    # Chooses action based on epsilon-greedy policy
    #
    def get_action(self, state):
        # Random action
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        # Exploit
        action_values = self.train_model.predict(state)

        # Max value is the action
        return np.argmax(action_values[0])

    #
    # Trains model using a random batch of selected experiences the memory
    #
    def train(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            if not done:
                target = (reward + self.gamma * np.amax(self.target_model.predict(next_state)))
            else:
                target = reward

            # Construct the target vector as follows:
            # 1. Use the current model to output the Q-value predictions
            target_f = self.train_model.predict(state)
            # 2. Rewrite the chosen action value with the computed target
            target_f[0][action] = target
            # 3. Use vectors in the objective computation
            self.train_model.fit(state, target_f, epochs=1, verbose=0)

        self.epsilon = max(min_epsilon, epsilon * decay)

    #
    # Sets the target model parameters to the current model parameters
    #
    def update_target_model(self):
        self.target_model.set_weights(self.train_model.get_weights())


# Helpful preprocessing taken from github.com/ageron/tiny-dqn
def process_frame(obs):
    img = obs[1:176:2, ::2] # crop and downsize
    img = img.mean(axis=2) # to greyscale
    img = (img - 128) / 128 - 1 # normalize from -1. to 1.
    return img.reshape(75, 300, 1)

def blend_images(images, blend):
    avg_image = np.expand_dims(np.zeros((75, 300, 1), np.float64), axis=0)

    for image in images:
        avg_image += image

    if len(images) < blend:
        return avg_image / len(images)
    else:
        return avg_image / blend


if __name__ == "__main__":
    env = gym.make('ChromeDino-v0')
    state = env.reset()

    state_shape = env.observation_space.shape # (150, 600, 3)
    state_shape = (75, 300, 1)
    num_states = len(env.observation_space.sample()) # 150
    num_actions = env.action_space.n    # 2

    gamma = 0.99 #  decay rate of past observations original 0.99
    update_rate = 100

    max_experiences = 10000
    min_experiences = 100

    epsilon = 0.99    # starting value of epsilon
    min_epsilon = 0.1 # final value for epsilon
    decay = 0.9       # epsilon decay rate

    lr = 1e-2         # learning rate

    agent = DQN_Agent(state_shape, num_actions, gamma, epsilon, min_epsilon, decay, lr, update_rate, max_experiences, min_experiences)

    episodes = 1000
    rewards = 0
    total_time = 0
    batch_size = 8
    blend = 4        # Number of images to blend
    iter = 0

    for ep in range(episodes):

        state = process_frame(env.reset())
        images = deque(maxlen=blend)  # Array of images to be blended
        images.append(state)

        while True:
            iter += 1

            # Return the avg of the last 4 frames
            state = blend_images(images, blend)

            # Transition Dynamics
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)

            # Return the avg of the last 4 frames
            next_state = process_frame(next_state)
            images.append(next_state)
            next_state = blend_images(images, blend)

            if done:
                rewards += env.unwrapped.game.get_score()
                print("iteration: ", iter, "total reward: ", rewards, "epsilon: ", epsilon)
                env.reset()
                done = False
                rewards = 0

            reward += env.unwrapped.game.get_score()

            # Store sequence in replay memory
            agent.add_experience(state, action, reward, next_state, done)

            if len(agent.memory) > batch_size and len(agent.memory) > agent.min_experiences:
                print("Training...")
                agent.train(batch_size)
