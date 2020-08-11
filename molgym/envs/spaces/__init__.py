"""Choices for search spaces"""

from gym import Space


class AllMolecules(Space):
    """An observation space that consists of molecules in the QM9 dataset"""

    def sample(self):
        raise NotImplementedError('This design space does not support sampling')

    def contains(self, x):
        return True  # All molecules are valid
