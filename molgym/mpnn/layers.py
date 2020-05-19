"""Layers needed to create the MPNN model.

Taken from the ``tf2`` branch of the ``nfp`` code:

https://github.com/NREL/nfp/blob/tf2/examples/tf2_tests.ipynb
"""

"""Adapted from https://github.com/NREL/nfp/blob/tf2/examples/tf2_tests.ipynb"""

import tensorflow as tf
from tensorflow.keras import layers


class MessageBlock(layers.Layer):
    """Message passing layer for MPNNs

    Takes the state of an atom and bond, and updates them by passing messages between nearby neighbors.

    Following the notation of Gilmer et al., the message function sums all of the atom states from
    the neighbors of each atom and then updates the node state by adding them to the previous state.
    """

    def __init__(self, atom_dimension, **kwargs):
        """
        Args:
             atom_dimension (int): Number of features to use to describe each atom
        """
        super(MessageBlock, self).__init__(**kwargs)
        self.atom_bn = layers.BatchNormalization()
        self.bond_bn = layers.BatchNormalization()
        self.bond_update_1 = layers.Dense(2 * atom_dimension, activation='sigmoid', use_bias=False)
        self.bond_update_2 = layers.Dense(atom_dimension)
        self.atom_update = layers.Dense(atom_dimension, activation='sigmoid', use_bias=False)
        self.atom_dimension = atom_dimension

    def call(self, inputs):
        original_atom_state, original_bond_state, connectivity = inputs

        # Batch norm on incoming layers
        atom_state = self.atom_bn(original_atom_state)
        bond_state = self.bond_bn(original_bond_state)

        # Gather atoms to bond dimension
        target_atom = tf.gather(atom_state, connectivity[:, 0])
        source_atom = tf.gather(atom_state, connectivity[:, 1])

        # Update bond states with source and target atom info
        new_bond_state = tf.concat([source_atom, target_atom, bond_state], 1)
        new_bond_state = self.bond_update_1(new_bond_state)
        new_bond_state = self.bond_update_2(new_bond_state)

        # Update atom states with neighboring bonds
        source_atom = self.atom_update(source_atom)
        messages = source_atom * new_bond_state
        messages = tf.math.segment_sum(messages, connectivity[:, 0])

        # Add new states to their incoming values (residual connection)
        bond_state = original_bond_state + new_bond_state
        atom_state = original_atom_state + messages

        return atom_state, bond_state
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'atom_dimension': self.atom_dimension
        })
        return config


class GraphNetwork(layers.Layer):
    """Layer that implements an entire MPNN neural network

    Create shte message passing layers and also implements reducing the features of all nodes in
    a graph to a single feature vector for a molecule.

    The reduction to a single feature for an entire molecule is produced by summing a single scalar value
    used to represent each atom. We chose this reduction approach under the assumption the energy of a molecule
    can be computed as a sum over atomic energies."""

    def __init__(self, atom_classes, bond_classes, atom_dimension, num_messages, 
                 output_layer_sizes=None, **kwargs):
        """
        Args:
             atom_classes (int): Number of possible types of nodes
             bond_classes (int): Number of possible types of edges
             atom_dimension (int): Number of features used to represent a node and bond
             num_messages (int): Number of message passing steps to perform
             output_layer_sizes ([int]): Number of dense layers that map the atom state to energy
             dropout (float): Dropout rate
        """
        super(GraphNetwork, self).__init__(**kwargs)
        self.atom_embedding = layers.Embedding(atom_classes, atom_dimension, name='atom_embedding')
        self.bond_embedding = layers.Embedding(bond_classes, atom_dimension, name='bond_embedding')
        self.message_layers = [MessageBlock(atom_dimension) for _ in range(num_messages)]
        
        # Make the output MLP
        if output_layer_sizes is None:
            output_layer_sizes = []
        self.output_layers = [layers.Dense(s, activation='relu') for s in output_layer_sizes]
        self.output_layer_sizes = output_layer_sizes
        self.last_layer = layers.Dense(1)

    def call(self, inputs):
        atom_types, bond_types, node_graph_indices, connectivity = inputs

        # Initialize the atom and bond embedding vectors
        atom_state = self.atom_embedding(atom_types)
        bond_state = self.bond_embedding(bond_types)

        # Perform the message passing
        for message_layer in self.message_layers:
            atom_state, bond_state = message_layer([atom_state, bond_state, connectivity])

        # Sum over all atoms in a mol to form a single fingerprint
        mol_state = tf.math.segment_sum(atom_state, node_graph_indices)
        
        # Apply the MLP layers
        for layer in self.output_layers:
            mol_state = layer(mol_state)

        # Reduce mol to a single prediction
        return self.last_layer(mol_state)

    def get_config(self):
        config = super().get_config()
        config.update({
            'atom_classes': self.atom_embedding.input_dim,
            'bond_classes': self.bond_embedding.input_dim,
            'atom_dimension': self.atom_embedding.output_dim,
            'output_layer_sizes': self.output_layer_sizes,
            'num_messages': len(self.message_layers)
        })
        return config


class Squeeze(layers.Layer):
    """Wrapper over the tf.squeeze operation"""

    def __init__(self, axis=1, **kwargs):
        """
        Args:
            axis (int): Which axis to squash
        """
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.squeeze(inputs, axis=self.axis)

    def get_config(self):
        config = super().get_config()
        config['axis'] = self.axis
        return config


custom_objects = {
    'GraphNetwork': GraphNetwork,
    'MessageBlock': MessageBlock,
    'Squeeze': Squeeze
}
