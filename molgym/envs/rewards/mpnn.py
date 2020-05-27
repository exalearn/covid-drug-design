"""Rewards based on a MPNN"""
from typing import List

import numpy as np
import networkx as nx
from tensorflow.keras.models import Model

from molgym.envs.rewards import RewardFunction
from molgym.mpnn.data import convert_nx_to_dict


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

    def _call(self, graph: nx.Graph) -> float:
        # Convert the graph to dict format, and add in "node_graph_indices"
        entry = convert_nx_to_dict(graph, self.atom_types, self.bond_types)
        if entry['n_bond'] == 0:
            return self.big_value
        entry = dict((k, np.array(v)) for k, v in entry.items())
        entry['node_graph_indices'] = np.zeros((entry['n_atom'],))

        # Run the molecule as a batch
        output = self.model.predict_on_batch(entry)
        return float(output[0, 0])
