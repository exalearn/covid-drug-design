"""Utilities for using models to perform inference"""
from tensorflow.keras.models import Model
from graphsage.importing import create_inputs_from_nx
import networkx as nx
import numpy as np


# TODO (wardlt): Make a batch version


def run_inference(model: Model, graph: nx.Graph) -> float:
    """Run inference on a single graph

    Args:
        model (Model): Keras model to use for inference
        graph: Graph to use as input
    Return:
         (float) Predicted energy
    """

    # Compute the input
    batch = create_inputs_from_nx(graph)
    batch['node_graph_indices'] = np.zeros(shape=(batch['n_atom'],))
    batch['bond_graph_indices'] = np.zeros(shape=(batch['n_bond'],))
    batch = dict((k, np.array(v)) for k, v in batch.items())

    # Run the batch
    energies = model.predict_on_batch(batch)
    return energies[0, 0]
