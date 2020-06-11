"""Multi-objective reward functions"""
import logging
from typing import List

import networkx as nx

from molgym.envs.rewards import RewardFunction

logger = logging.getLogger(__name__)


class AdditiveReward(RewardFunction):
    """Adding several reward functions together

    If provided, each individual reward is normalized by a user-provided
    mean and scale factor (e.g., r' = (r - &mu;<sub>r</sub>) / &sigma;<sub>r</sub>)
    """

    def __init__(self, reward_functions: List[dict]):
        """
        Args:
            reward_functions ([dict]): List of reward function definitions. Each item is a dict with keys:
                - reward (RewardFunction): Reward function. Should define whether the reward is maximized or minimized
                - mean (float): Mean value for the reward. Default: 0
                - scale (float): Scale value for the reward. Default: 1
        """
        super().__init__(maximize=True)

        # Store the reward functions
        self._rewards: List[RewardFunction] = []
        self._means = []
        self._scales = []
        for r in reward_functions:
            assert 'reward' in r, "Reward definition must contain the reward function"
            self._rewards.append(r['reward'])
            self._means.append(r.get('mean', 0))
            self._scales.append(r.get('scale', 1))
            logger.info(f'Added {self._rewards[-1]} with mean {self._means[-1]: .2e} and scale {self._scales[-1]: .2e}')

    def _call(self, graph: nx.Graph) -> float:
        output = 0.
        for r, m, s in zip(self._rewards, self._means, self._scales):
            val = r(graph)
            if r.maximize:
                output += (val - m) / s
            else:
                output += (val + m) / s  # Val is equal to -1 * (reward)
        return output
