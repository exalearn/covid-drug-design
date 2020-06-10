"""Reward functions derived from RDKit"""
import sys
import os

import networkx as nx
from rdkit.Chem import Crippen, QED

from molgym.envs.rewards import RewardFunction
from molgym.utils.conversions import convert_nx_to_rdkit

# Load the SA score
env_path = os.path.dirname(os.path.dirname(sys.executable))
_sa_score_path = f"{env_path}/share/RDKit/Contrib/SA_Score/sascorer.py"
if not os.path.isfile(_sa_score_path):
    raise ValueError('SA_scorer file not found. You must edit the above lines to point to the right place. Sorry!')
sys.path.append(os.path.dirname(_sa_score_path))
from sascorer import calculateScore


class LogP(RewardFunction):
    """Water/octanol partition coefficient"""

    def _call(self, graph: nx.Graph) -> float:
        mol = convert_nx_to_rdkit(graph)
        return Crippen.MolLogP(mol)


class QEDReward(RewardFunction):
    """Quantitative measure of uncertainty"""

    def _call(self, graph: nx.Graph) -> float:
        mol = convert_nx_to_rdkit(graph)
        return QED.qed(mol)


class SAScore(RewardFunction):
    """Synthesis accessibility score

    Smaller values indicate greater "synthesizability" """

    def _call(self, graph: nx.Graph) -> float:
        mol = convert_nx_to_rdkit(graph)
        return calculateScore(mol)
