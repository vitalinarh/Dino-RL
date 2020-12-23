try:
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
    import cv2
    import display
    import argparse
    import sys
except:
    import install_requirements  # install packages
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
    import argparse
    import sys


class DQN_Agent:
    #
    # Initialize attributes and construct CNN train_model and target_model
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
        self.target_model.set_weights(self.train_model.get_weights())
        self.train_model.summary()

    #
    # Constructs model
    #
    def _build_model(self):

        model = Sequential()

        # Convolutional layers

        model.add(Conv2D(16, (8, 8), strides=4, padding='same', input_shape=self.state_shape))
        model.add(Activation('relu'))

        model.add(Conv2D(32, (4, 4), strides=2, padding='same'))
        model.add(Activation('relu'))

        model.add(Conv2D(32, (3, 3), strides=1, padding='same'))
        model.add(Activation('relu'))
        model.add(Flatten())

        # FC Layers
        model.add(Dense(self.action_size, activation='linear'))

        model.compile(loss='mse', optimizer=Adam(learning_rate=self.lr))

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
            action = np.random.choice(self.action_size)
            return action

        # Exploit
        action_values = self.train_model.predict(state)
        # Max value is the action
        action = np.argmax(action_values[0])
        return action

    #
    # Trains model using a random batch of selected experiences in the memory
    #
    def train(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.target_model.predict(next_state)[0]))

            target_f = self.train_model.predict(state)
            target_f[0][action] = target
            self.train_model.fit(state, target_f, epochs=1, verbose=0)

    #
    # Set the target model parameters to the current model parameters
    #
    def update_target_model(self):
        self.target_model.set_weights(self.train_model.get_weights())

    #
    # Save parameters of the trained models
    #
    def save_network(self, name):
        self.train_model.save_weights(name)

    #
    # Load parametres of the trained model
    #
    def load_network(self, name):
        self.train_model.load_weights(name)


# Preprocessing taken from github.com/ageron/tiny-dqn
def process_frame(obs):
    #display.display(obs, "original")
    img = obs[40:, 20:170]     # crop and downsize
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img = img.mean(axis=2)      # to greyscale
    img = (img - 128) / 128 - 1 # normalize from -1. to 1.
    #display.display(img, "cropped")
    return img.reshape(110, 150, 1)

def blend_images(images, blend):
    avg_image = np.expand_dims(np.zeros((110, 150, 1), np.float64), axis=0)

    for image in images:
        avg_image += image

    if len(images) < blend:
        return avg_image / len(images)
    else:
        return avg_image / blend


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Dino ML Agent')
    parser.add_argument("-train", action='store_const', const='train', dest="argv" ,help='Train an agent. Previous progress saved in folder "models"')
    parser.add_argument("-newTrain", action='store_const', const='newTrain' ,dest="argv", help='Train an new agent. Previous agent weights will be deleted.')
    parser.add_argument("-test", action='store_const', const='test' ,dest="argv", help='Use the weights saved in the folder "models" for the agent.')

    args = parser.parse_args()

    # check args
    if args.argv=="train" or args.argv=="test":
        pass
    elif args.argv=="newTrain":
        os.system('rm -r models')
        os.system('rm logs.txt')
    else:
        print("usage: dino.py [-h] [-train] [-newTrain] [-test]")
        sys.exit()

    env = gym.make('ChromeDino-v0')
    state = env.reset()

    state_shape = env.observation_space.shape # (150, 600, 3)
    state_shape = (110, 150, 1)                # downsample of the original state size
    num_actions = env.action_space.n          # 2

    gamma = 0.99 #  decay rate of past observations original 0.99
    update_rate = 1000
    train_rate = 500

    max_experiences = 15000
    min_experiences = 1000

    epsilon = 0.99        # starting value of epsilon
    min_epsilon = 0.1    # final value for epsilon
    decay = 0.99       # epsilon decay rate

    lr = 1e-2         # learning rate

    episodes = 10000
    rewards = 0
    total_time = 0
    batch_size = 32
    blend = 4        # Number of images to blend

    total_steps = 0


    
    agent = DQN_Agent(state_shape, num_actions, gamma, epsilon, min_epsilon, decay, lr, update_rate, max_experiences, min_experiences)

    if args.argv=="test":
        agent.load_network("models/agent")

    for ep in range(episodes):

        iter = 0
        state = process_frame(env.reset())
        images = deque(maxlen=blend)  # Array of images to be blended
        images.append(state)
        next_state, reward, done, info = env.step(1)

        rewards = 0
        done = False

        while True:
            iter += 1
            total_steps += 1
            reward = 0

            # Every update_rate timesteps we update the target network parameters
            if total_steps % agent.update_rate == 0:
                agent.update_target_model()

            # Return the average of the last 4 frames
            state = blend_images(images, blend)

            # Transitions from one state to the next through the chosen action
            if iter < 40:
                action = 0
            else:
                action = agent.get_action(state)

            next_state, reward, done, info = env.step(action)

            # Return the avg of the last 4 frames
            next_state = process_frame(next_state)
            images.append(next_state)
            next_state = blend_images(images, blend)

            if done:
                agent.epsilon = max(agent.min_epsilon, agent.epsilon * agent.decay)
                rewards = env.unwrapped.game.get_score()

                print("episode: ", ep,
                      "iteration: ", total_steps,
                      "total reward: ", rewards,
                      "epsilon: ", agent.epsilon,
                      "experiences:", len(agent.memory))
                if args.argv=="train" or args.argv=="newTrain":
                    f = open("logs.txt", "a")
                    L = "%d %d %d %f\n" % (ep+1, rewards, total_steps, agent.epsilon)
                    f.writelines(L)
                    f.close()
                    if len(agent.memory) > batch_size:
                        print("Training...")
                        agent.train(batch_size)
                        agent.save_network("models/agent")

                break

            #reward += env.unwrapped.game.get_score()

            # Store sequence in replay memory
            if args.argv=="train" or args.argv=="newTrain":
                agent.add_experience(state, action, reward, next_state, done)

            state = next_state
