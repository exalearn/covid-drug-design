import random
import logging
from typing import Optional

import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Lambda, Subtract, Concatenate
from tensorflow.keras import backend as K

from molgym.agents.preprocessing import MorganFingerprints
from molgym.envs.simple import Molecule

logger = logging.getLogger(__name__)


def _q_target_value(inputs, gamma=0.99):
    """Function to compute the target value for Q learning"""
    reward, v_tp1, done = inputs
    return reward + gamma * (1.0 - done) * v_tp1


class DQNFinalState:
    """Implementation of Deep Q Learning that uses the final state after applying an action as input

    Q is typically defined as a function of (state, action), written as Q(s, a).
    Here, we define a new state, s', as the result of applying action a to state s
    and use s' as the input to Q.

    Follows the implementation described by `Zhou et al. <http://www.nature.com/articles/s41598-019-47148-x>`_.
    """

    def __init__(self, env: Molecule, preprocessor: MorganFingerprints, epsilon=1.0):
        """
        Args:
            epsilon (float): Exploration factor
            preprocessor (MorganFingerprints): Tool to compute Morgan fingerprints for each molecule
        """
        self.env = env
        self.preprocessor = preprocessor
        self.memory = deque(maxlen=2000)

        # Hyper-parameters
        self.gamma = 0.995    # discount rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32

        # Create the model
        self._build_model()

    def _huber_loss(self, target, prediction):
        error = prediction - target
        return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)

    def _build_model(self):
        # Get the shape of the environment
        fingerprint_size = self.preprocessor.length

        predict_actions_input = Input(batch_shape=(None, fingerprint_size), name='single_action')
        train_action_input = Input(batch_shape=(self.batch_size, fingerprint_size),
                                   name='batch_action')
        reward_input = Input(batch_shape=(self.batch_size, 1), name='rewards')
        done_input = Input(batch_shape=(self.batch_size, 1), name='done')
        new_actions = [Input(batch_shape=(None, fingerprint_size), name=f'next_actions_{i}')
                       for i in range(self.batch_size)]

        # Squeeze the train action and reward input
        squeeze = Lambda(K.squeeze, arguments={'axis': 1}, name='squeezer')
        reward = squeeze(reward_input)
        done = squeeze(done_input)

        # Define the Q network. Note that we use a `Network` rather than model because
        #   this model is not trained
        # - Takes a list of actions as input
        # - Produces the value of each action as output
        # TODO (wardlt): Allow users to specify a different architecture

        def make_q_network(input_shape, name=None):
            inputs = Input(batch_shape=input_shape, name=f'{name}_input')
            h1 = Dense(24, activation='relu')(inputs)
            h2 = Dense(48, activation='relu')(h1)
            h3 = Dense(24, activation='relu')(h2)
            output = Dense(1, activation='linear')(h3)
            return Model(inputs=inputs, outputs=output, name=name)

        q_t = make_q_network((None, fingerprint_size), name='q_t')
        q = q_t(predict_actions_input)
        self.action_network = Model(inputs=predict_actions_input, outputs=q)

        # Make the training network
        # Part 1: Computing estimated value of the next state
        #  Set as the maximum Q for any action from that next state
        #  Note: This Q network is not updated by the optimizer. Instead, it is
        #   periodically updated with the weights from `q_t`, which is being updated
        q_tp1 = make_q_network((None, fingerprint_size), name='q_tp1')
        q_tp1.trainable = False
        max_layer = Lambda(K.max, arguments={'axis': 0}, name='max_layer')
        q_values = [max_layer(q_tp1(action)) for action in new_actions]
        v_tp1 = Concatenate(name='v_tp1')(q_values)

        # Part 2: Define the target function, the measured reward of a state
        #   plus the estimated value of the next state (or zero if this state is terminal)
        target = Lambda(_q_target_value, name='target', arguments={'gamma': self.gamma})\
            ([reward, v_tp1, done])

        # Part 3: Define the error signal
        q_t_train = q_t(train_action_input)
        q_t_train = Lambda(K.reshape, arguments={'shape': (self.batch_size,)},
                           name='squeeze_q')(q_t_train)
        error = Subtract(name='error')([q_t_train, target])
        error = Lambda(K.reshape, arguments={'shape': (self.batch_size, 1)},
                       name='wrap_error')(error)

        self.train_network = Model(
            inputs=[train_action_input, done_input, reward_input] + new_actions,
            outputs=error)

        # Add the optimizer
        self.optimzer = tf.keras.optimizers.Adam()

    def remember(self, state, action, reward, next_state, next_actions, done):
        # Save the actions as features, we no longer need to know they are molecules
        action_features = self.preprocessor.get_features([action])[0]
        if len(next_actions) > 0:
            next_actions_features = self.preprocessor.get_features(next_actions)
        else:
            next_actions_features = []
        self.memory.append((action_features, reward, next_actions_features, done))

    def action(self):
        """Choose the next action"""
        # Get the actions as SMILES
        actions = self.env.action_space.get_possible_actions()

        if np.random.rand() <= self.epsilon:
            action_ix = random.randrange(self.env.action_space.n)
        else:
            # Invoke the action network, which gives the action with the highest reward
            actions_features = self.preprocessor.get_features(actions)
            action_scores = self.action_network.predict(actions_features)
            action_ix = np.argmax(action_scores)
        return actions[action_ix]

    def update_target_q_network(self):
        """Updates the Q function used to define the target to use the current Q network"""

        q_weights = self.action_network.get_layer('q_t').get_weights()
        self.train_network.get_layer('q_tp1').set_weights(q_weights)

    def train(self) -> Optional[float]:
        """Train model on a batch of data from the memory

        Returns:
            loss (float): Current loss
        """

        # Check if we have enough data
        if len(self.memory) < self.batch_size:
            return

        # Get a minibatch
        actions, rewards, next_actions, done = zip(*random.sample(self.memory, self.batch_size))

        # Convert inputs to numpy arrays
        actions = np.array(actions, dtype=K.floatx())
        rewards = np.array(rewards, dtype=K.floatx())
        next_actions = [np.array(a, dtype=K.floatx()) for a in next_actions]
        done = np.array(done, dtype=K.floatx())

        # Give bogus moves to those that are done and lack next moves
        #  Needed to give the proper input shape to the model
        for i, (na, d) in enumerate(zip(next_actions, done)):
            if na.shape == (0,):
                if not d:
                    raise RuntimeError('Found a move that is not terminal, yet has no next actions')
                next_actions[i] = np.zeros((1, self.action_network.input_shape[1]))

        # Compute the error signal between the data
        with tf.GradientTape() as tape:
            error = self.train_network.predict_on_batch([actions, done, rewards] + list(next_actions))
            loss = tf.reduce_sum(tf.pow(error, 2, name='loss_pow'), name='loss_sum')
        gradients = tape.gradient(loss, self.train_network.trainable_variables)
        self.optimzer.apply_gradients(zip(gradients, self.train_network.trainable_variables))
        return float(loss.numpy())

    def epsilon_adj(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, path):
        """Load the weights of the model
        
        Args:
            path (str): Path to a file holding the weights
        """
        # Load in the training network
        self.train_network.load_weights(path)

        # Use it to set the weights of the "action network"
        q_weights = self.train_network.get_layer('q_t').get_weights()
        self.action_network.get_layer('q_t').set_weights(q_weights)

    def save(self, path):
        """Save the model state

        Args:
            path (str): Path to save weights
        """
        self.train_network.save_weights(path)
