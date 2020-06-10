"""Utilities for computing ML-ready inputs from a series of molecules"""
from typing import List, Any

import numpy as np
import networkx as nx
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from molgym.utils.conversions import convert_nx_to_rdkit


def compute_morgan_fingerprints(graph: nx.Graph, fingerprint_length: int, fingerprint_radius: int):
    """Get Morgan Fingerprint of a specific SMILES string.

    Adapted from: <https://github.com/google-research/google-research/blob/
    dfac4178ccf521e8d6eae45f7b0a33a6a5b691ee/mol_dqn/chemgraph/dqn/deep_q_networks.py#L750>

    Args:
      graph (nx.Graph): The molecule as a networkx graph
      fingerprint_length (int): Bit-length of fingerprint
      fingerprint_radius (int): Radius used to compute fingerprint
    Returns:
      np.array. shape = [hparams.fingerprint_length]. The Morgan fingerprint.
    """
    if graph is None:  # No smiles string
        return np.zeros((fingerprint_length,))
    molecule = convert_nx_to_rdkit(graph)
    if molecule is None:  # Invalid smiles string
        return np.zeros((fingerprint_length,))

    # Compute the fingerprint
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(
        molecule, fingerprint_radius, fingerprint_length)
    arr = np.zeros((1,))

    # ConvertToNumpyArray takes ~ 0.19 ms, while
    # np.asarray takes ~ 4.69 ms
    DataStructs.ConvertToNumpyArray(fingerprint, arr)
    return arr


class MoleculePreprocessor:
    """Base class for a tool to generate molecular features"""

    def get_features(self, graphs: List[nx.Graph]) -> Any:
        """Create features ready for use in a the models within an agent"""
        raise NotImplementedError()


class MorganFingerprints(MoleculePreprocessor):
    """Compute the Morgan fingerprints for a series of molecules"""

    def __init__(self, length: int = 2048, radius: int = 3):
        self.length = length
        self.radius = radius

    def get_features(self, smiles: List[nx.Graph]) -> Any:
        return np.vstack([compute_morgan_fingerprints(m, self.length, self.radius) for m in smiles])
