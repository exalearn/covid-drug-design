"""Different chocies for reward functions"""

from rdkit import Chem
from rdkit.Chem import Crippen


class RewardFunction:
    """Base class for molecular reward functions"""

    def __call__(self, smiles: str) -> float:
        """Compute the reward for a certain molecule

        Args:
            smiles (str): SMILES string of a molecule
        Returns:
            (float) Reward
        """
        raise NotImplementedError()


class LogP(RewardFunction):

    def __call__(self, smiles: str) -> float:
        mol = Chem.MolFromSmiles(smiles)
        return Crippen.MolLogP(mol)
