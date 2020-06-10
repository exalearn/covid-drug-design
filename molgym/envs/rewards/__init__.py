"""Different choices for reward functions"""

import networkx as nx


class RewardFunction:
    """Base class for molecular reward functions"""

    def __init__(self, maximize: bool = True):
        """
        Args:
            maximize (bool): Whether to maximize the objective function
        """
        self.maximize = maximize

    def __call__(self, graph: nx.Graph) -> float:
        """Compute the reward for a certain molecule

        Args:
            graph (str): NetworkX graph form of the molecule
        Returns:
            (float) Reward
        """
        reward = self._call(graph)
        if self.maximize:
            return reward
        return -1 * reward

    def _call(self, graph: nx.Graph) -> float:
        """Compute the reward for a certain molecule

        Private version of the method. The public version
        handles switching signs if needed

        Args:
            graph (str): NetworkX graph form of the molecule
        Returns:
            (float) Reward
        """
        raise NotImplementedError()
