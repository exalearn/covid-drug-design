"""Rewards based on a MPNN"""
from typing import List

import numpy as np
import networkx as nx
from tensorflow.keras.models import Model

from molgym.envs.rewards import RewardFunction
from molgym.mpnn.data import convert_nx_to_dict


class MPNNReward(RewardFunction):

    def __init__(self, model: Model, atom_types: List[int], bond_types: List[str]):
        """
        Args:
            model: Keras MPNN model (trained using the tools in this package)
            atom_types: List of known atomic types
            bond_types: List of known bond types
        """
        super().__init__()
        self.model = model
        self.atom_types = atom_types
        self.bond_types = bond_types

    def __call__(self, graph: nx.Graph) -> float:
        # Convert the graph to dict format, and add in "node_graph_indices"
        entry = convert_nx_to_dict(graph, self.atom_types, self.bond_types)
        if entry['n_bond'] == 0:
            return 0
        entry = dict((k, np.array(v)) for k, v in entry.items())
        entry['node_graph_indices'] = np.zeros((entry['n_atom'],))

        # Run the molecule as a batch
        output = self.model.predict_on_batch(entry)
        return float(output[0, 0])
