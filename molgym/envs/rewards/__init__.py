"""Different choices for reward functions"""

import networkx as nx
from rdkit import Chem
from rdkit.Chem import Crippen

from molgym.utils.conversions import convert_nx_to_rdkit


class RewardFunction:
    """Base class for molecular reward functions"""

    def __call__(self, graph: nx.Graph) -> float:
        """Compute the reward for a certain molecule

        Args:
            graph (str): NetworkX graph form of the molecule
        Returns:
            (float) Reward
        """
        raise NotImplementedError()


class LogP(RewardFunction):

    def __call__(self, graph: nx.Graph) -> float:
        mol = convert_nx_to_rdkit(graph)
        return Crippen.MolLogP(mol)
