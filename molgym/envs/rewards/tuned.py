"""Hand-tuned objective functions for drug design"""
from typing import Tuple

from rdkit.Chem import Descriptors
import networkx as nx
import numpy as np

from molgym.envs.rewards import RewardFunction
from molgym.envs.rewards.mpnn import MPNNReward
from molgym.envs.rewards.oneshot import OneShotScore
from molgym.utils.conversions import convert_nx_to_rdkit


class LogisticCombination(RewardFunction):
    """Combines several objectives together:

        1. *Molecular Weight*: Should be between 200 and 600 Dalton
        2. *pIC50*: 8 is our target minimum, higher is better
        3. *Similarity to Training Set*: We have a one-hot encoding model that
           produces the similarity to some target molecules

    All rewards exist on different scales, so we use logistic functions to make
    them exist on [0, 1] where 1 is the best value.
    """

    def __init__(self, pic50_reward: MPNNReward, similarity_reward: OneShotScore,
                 target_pic50: int = 8, pic50_tol: float = 0.5,
                 mw_range: Tuple[int, int] = (200, 600), mw_tol: float = 25):
        """
        Args:
            pic50_reward: Reward function for computing pIC50
            similarity_reward: Reward function for computing similiarity to target molecules
            target_pic50: Target pIC50 value (default: 8)
            pic50_tol: Controls how quickly reward function decays below target pIC50 and
                increases above the target (default: 0.5, larger means slower decay)
            mw_range: Target molecular weight range. Unit: Da (default: 200 - 600 Da)
            mw_tol: Controls how quickly reward decays as MW moves in/out of the target range.
                (default: 25, larger means slower decay)
        """
        super().__init__(maximize=True)

        self.pic50_reward = pic50_reward
        self.similarity_reward = similarity_reward
        self.target_pic50 = target_pic50
        self.pic50_tol = pic50_tol
        self.mw_range = mw_range
        self.mw_tol = mw_tol

    def _call(self, graph: nx.Graph) -> float:
        # Compute all the rewards
        mol = convert_nx_to_rdkit(graph)
        mw = Descriptors.MolWt(mol)
        pic50 = self.pic50_reward(graph)
        sim = self.similarity_reward(graph)

        # Normalize them
        mw_score = 1.0 / (
            (1 + np.exp((self.mw_range[0] - mw) /self.mw_tol)) *
            (1 + np.exp((mw - self.mw_range[1]) / self.mw_tol))
        )
        pic50_score = 1.0 / (1 + np.exp((8 - pic50) / self.pic50_tol))
        # Sim is already between [0, 1]

        # Combine them
        return sim + mw_score + pic50_score
