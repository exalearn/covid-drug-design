"""Rate a molecule as similar to a training set or not"""
from typing import List

import networkx as nx
from tensorflow.keras.models import Model

from molgym.envs.rewards.mpnn import MPNNReward
from molgym.mpnn.data import convert_nx_to_dict, create_batches_from_objects


class OneShotScore(MPNNReward):
    """Score a molecule as similar to an existing distribution
    using a model created by one-shot learning"""

    def __init__(self, model: Model, atom_types: List[int], bond_types: List[str],
                 target_molecules: List[nx.Graph], batch_size: int = 256, maximize=True):
        """
        Args:
            model: Keras MPNN model for one-shot learning
            atom_types: List of known atomic types
            bond_types: List of known bond types
            target_molecules: Set of molecules to compare
        """
        super().__init__(model, atom_types, bond_types, maximize, big_value=0 if maximize else 1)

        # Convert the target molecules into batches
        target_dicts = [convert_nx_to_dict(g, self.atom_types, self.bond_types) for g in target_molecules]
        self.batch_size = batch_size
        target_batches_temp = create_batches_from_objects(target_dicts, batch_size=batch_size)

        # Append a "_l" to the inputs for each of the batches
        self.target_molecules_length = len(target_molecules)
        self.target_batches = []
        for b in target_batches_temp:
            new_dict = dict((f'{k}_l', v) for k, v in b.items())
            self.target_batches.append(new_dict)

    def _call(self, graph: nx.Graph) -> float:
        # Convert the graph to dict format, and add in "node_graph_indices"
        entry = convert_nx_to_dict(graph, self.atom_types, self.bond_types)
        if entry['n_bond'] == 0:
            return self.big_value

        # Make a set of batches for the "right side", merge them with the left-side batches
        batches_r = create_batches_from_objects([entry]*self.target_molecules_length, self.batch_size)
        comparisons = []
        for batch_l, batch_r in zip(self.target_batches, batches_r):
            new_dict = dict((f'{k}_r', v) for k, v in batch_r.items())
            new_dict.update(batch_l)
            comparisons.append(new_dict)

        # Compute the maximum similarity
        output = 0
        for batch in comparisons:
            preds = self.model.predict_on_batch(batch)
            output = max(preds.numpy().max(), output)
        return float(output)  # Convert from float32 (not JSON-serializable)
