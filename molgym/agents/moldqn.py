import random
import logging
from typing import Optional, Iterable, Tuple, Any

import numpy as np
import pandas as pd
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Model, model_from_config
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

    def __init__(self, env: Molecule, preprocessor: MorganFingerprints, gamma: float = 0.9,
                 batch_size: int = 32, epsilon: float = 1.0, q_network_dense: Iterable[int] = (24, 48, 24),
                 epsilon_decay: float = 0.995, memory_size: int = 2000):
        """
        Args:
            env (Molecule): Molecule environment
            gamma (float): Discount rate for
            batch_size (int): Size of each training batch
            epsilon (float): Exploration rate, beginning
            preprocessor (MorganFingerprints): Tool to compute Morgan fingerprints for each molecule
            q_network_dense ([int]): Number of units in each hidden layer for the Q networks
            epsilon_decay (float): Fraction to decay epsilon after each epoch
            memory_size (int): Number of records to hold in memory
        """
        self.env = env
        self.preprocessor = preprocessor
        self.memory = deque(maxlen=memory_size)

        # Hyper-parameters
        self.gamma = gamma  # discount rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = 0.10
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.q_network_dense = q_network_dense

        # Create the model
        self._build_model()

    def __getstate__(self):
        output = self.__dict__.copy()

        # Store the weights of the networks and not the network object
        for i in ['action_network', 'train_network']:
            output[f'{i}_weights'] = output[i].get_weights()
            del output[i]
        return output

    def __setstate__(self, state: dict):
        # Make an internal copy to avoid overwriting
        state = state.copy()

        # Remove the weights from the state
        _networks = ['action_network', 'train_network']
        weights = dict((n, state.pop(f'{n}_weights')) for n in _networks)

        # Set the rest of the state
        self.__dict__.update(state)

        # Rebuild networks and set the state
        self._build_model()
        for n in _networks:
            getattr(self, n).set_weights(weights.pop(n))

    def _huber_loss(self, target, prediction):
        error = prediction - target
        return tf.reduce_mean(tf.sqrt(1+tf.math.square(error)) - 1, axis=-1)

    def _build_model(self):
        # Get the shape of the environment
        fingerprint_size = self.preprocessor.length

        predict_actions_input = Input(batch_shape=(None, fingerprint_size), name='single_action')
        train_action_input = Input(batch_shape=(self.batch_size, fingerprint_size),
                                   name='batch_action')
        reward_input = Input(batch_shape=(self.batch_size, 1), name='rewards')
        done_input = Input(batch_shape=(self.batch_size, 1), name='done')
        next_actions = Input(batch_shape=(None, fingerprint_size), name=f'next_actions')
        next_actions_id = Input(batch_shape=(None,), name='next_action_batch_id', dtype=tf.int32)

        # Squeeze the train action and reward input
        squeeze = Lambda(K.squeeze, arguments={'axis': 1}, name='squeezer')
        reward = squeeze(reward_input)
        done = squeeze(done_input)

        # Define the Q network. Note that we use a `Network` rather than model because
        #   this model is not trained
        # - Takes a list of actions as input
        # - Produces the value of each action as output

        def make_q_network(input_shape, name=None):
            inputs = Input(batch_shape=input_shape, name=f'{name}_input')
            h = inputs
            for n in self.q_network_dense:
                h = Dense(n, activation='relu')(h)
            output = Dense(1, activation='linear')(h)
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
        next_actions_q = q_tp1(next_actions)
        max_layer = Lambda(lambda x: tf.math.segment_max(*x), name='v_tp1')
        v_tp1 = max_layer([next_actions_q, next_actions_id])

        # Part 2: Define the target function, the measured reward of a state
        #   plus the estimated value of the next state (or zero if this state is terminal)
        target = Lambda(_q_target_value, name='target',
                        arguments={'gamma': self.gamma})([reward, v_tp1, done])

        # Part 3: Define the error signal
        q_t_train = q_t(train_action_input)
        q_t_train = Lambda(K.reshape, arguments={'shape': (self.batch_size,)},
                           name='squeeze_q')(q_t_train)
        error = Subtract(name='error')([q_t_train, target])

        self.train_network = Model(
            inputs=[train_action_input, done_input, reward_input, next_actions, next_actions_id],
            outputs=error)

        # Add the optimizer
        self.optimizer = tf.keras.optimizers.Adam()

    def remember(self, state, action, reward, next_state, next_actions, done):
        # Save the actions as features, we no longer need to know they are molecules
        action_features = self.preprocessor.get_features([action])[0]
        if len(next_actions) > 0:
            next_actions_features = self.preprocessor.get_features(next_actions)
        else:
            next_actions_features = []
        self.memory.append((action_features, reward, next_actions_features, done))

    def action(self) -> Tuple[Any, float, bool]:
        """Choose the next action

        Returns:
            - Selected next step
            - (float) Predicted Q for the next step
            - (bool) Whether the move was random
        """
        # Get the actions as SMILES
        actions = self.env.action_space.get_possible_actions()

        # Invoke the action network, which gives the "q" for each action
        actions_features = self.preprocessor.get_features(actions)
        actions_features = tf.convert_to_tensor(actions_features)
        action_scores = self.action_network.predict(actions_features)

        # Get the next
        random_move = np.random.rand() <= self.epsilon
        if random_move:
            action_ix = random.randrange(self.env.action_space.n)
        else:
            action_ix = np.argmax(action_scores)
        q = action_scores[action_ix][0]
        return actions[action_ix], q, random_move

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

        # Stack the "next actions" into a single array, adding dummies where needed
        next_actions = [na if np.size(na) > 0 else np.zeros((1, self.preprocessor.length))
                        for na in next_actions]
        next_action_size = [x.shape[0] for x in next_actions]

        # Convert inputs to numpy arrays
        actions = tf.convert_to_tensor(actions, dtype=K.floatx())
        rewards = tf.convert_to_tensor(rewards, dtype=K.floatx())
        next_actions = tf.convert_to_tensor(np.vstack(next_actions), dtype=K.floatx())
        next_actions_ids = tf.convert_to_tensor(np.repeat(range(self.batch_size), next_action_size))
        done = tf.convert_to_tensor(done, dtype=K.floatx())

        # Give bogus moves to those that are done and lack next moves
        #  Needed to give the proper input shape to the model
        for i, (na, d) in enumerate(zip(next_actions, done)):
            if na.shape == (0,):
                if not d:
                    raise RuntimeError('Found a move that is not terminal, yet has no next actions')
                next_actions[i] = tf.zeros((1, self.action_network.input_shape[1]))

        # Compute the error signal between the data
        with tf.GradientTape() as tape:
            error = self.train_network.predict_on_batch([actions, done, rewards, next_actions, next_actions_ids])
            loss = tf.reduce_mean(tf.sqrt(1 + tf.math.square(error)) - 1)  # Huber Loss
        gradients = tape.gradient(loss, self.train_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.train_network.trainable_variables))
        return float(loss.numpy())

    def epsilon_adj(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load_model(self, path):
        """Load the weights of the model
        
        Args:
            path (str): Path to a file holding the weights
        """
        # Load in the training network
        self.train_network.load_weights(path)

        # Use it to set the weights of the "action network"
        q_weights = self.train_network.get_layer('q_t').get_weights()
        self.action_network.get_layer('q_t').set_weights(q_weights)

    def save_model(self, path):
        """Save the model state

        Args:
            path (str): Path to save weights
        """
        self.train_network.save_weights(path)

    def save_data(self, path):
        """Save the training data for the model

        Saves the data in JSON-LD format.

        Args:
            path (str): Path to output data
        """

        data = pd.DataFrame(self.memory, columns=['actions', 'rewards', 'next_actions', 'done'])
        data.to_json(path, orient='records', lines=True)
