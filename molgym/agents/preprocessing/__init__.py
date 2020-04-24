"""Utilities for computing ML-ready inputs from a series of molecules"""
from typing import List, Any

import numpy as np

from molgym.envs.utils import compute_morgan_fingerprints


class MoleculePreprocessor:
    """Base class for a tool to generate molecular features"""

    def get_features(self, smiles: List[str]) -> Any:
        """Create features ready for use in a the models within an agent"""
        raise NotImplementedError()


class MorganFingerprints(MoleculePreprocessor):
    """Compute the Morgan fingerprints for a series of molecules"""

    def __init__(self, length: int = 512, radius: int = 3):
        self.length = length
        self.radius = radius

    def get_features(self, smiles: List[str]) -> Any:
        return np.vstack([compute_morgan_fingerprints(m, self.length, self.radius) for m in smiles])
