"""Rewards based on a MPNN"""
from typing import List

import numpy as np
import networkx as nx
import tensorflow as tf
from tensorflow.keras.models import Model

from molgym.envs.rewards import RewardFunction
from molgym.mpnn.data import convert_nx_to_dict
from molgym.mpnn.layers import custom_objects


class MPNNReward(RewardFunction):

    def __init__(self, model: Model, atom_types: List[int], bond_types: List[str],
                 maximize: bool = True, big_value: float = 100.):
        """
        Args:
            model: Keras MPNN model (trained using the tools in this package)
            atom_types: List of known atomic types
            bond_types: List of known bond types
            maximize: Whether to maximize or minimize the target function
            big_value: Stand-in value to use for compounds the MPNN fails on
        """
        super().__init__(maximize)
        self.model = model
        self.atom_types = atom_types
        self.bond_types = bond_types
        self.big_value = abs(big_value)
        if self.maximize:
            self.big_value *= -1

    def __getstate__(self):
        state = self.__dict__.copy()
        
        # Convert the model to a JSON description and weights
        state['model_weights'] = self.model.get_weights()
        state['model'] = self.model.to_json()

        return state

    def __setstate__(self, state):
        state = state.copy()

        # Convert the MPNN model back to a Keras object
        state['model'] = tf.keras.models.model_from_json(state['model'], custom_objects=custom_objects)
        state['model'].set_weights(state.pop('model_weights'))

        self.__dict__.update(state)

    def _call(self, graph: nx.Graph) -> float:
        # Convert the graph to dict format, and add in "node_graph_indices"
        entry = convert_nx_to_dict(graph, self.atom_types, self.bond_types)
        if entry['n_bond'] == 0:
            return self.big_value
        entry = dict((k, tf.convert_to_tensor(v)) for k, v in entry.items())
        entry['node_graph_indices'] = tf.zeros((entry['n_atom'],))

        # Run the molecule as a batch
        output = self.model.predict_on_batch(entry)
        return float(output[0, 0])
