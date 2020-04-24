from gym import Space
from rdkit import Chem
from rdkit.Chem import Draw
from functools import partial
import logging
import gym

from .actions import MoleculeActions
from .spaces import AllMolecules
from .rewards import LogP, RewardFunction
from .utils import compute_morgan_fingerprints

logger = logging.getLogger(__name__)


def compile_smiles(mol) -> str:
    """Compute the InCHi string of a RDKit molecule object

    Args:
        mol (Mol): RDKit molecule
    Returns:
        (str) InChI string
    """
    return Chem.MolToSmiles(mol)


class Molecule(gym.Env):
    """Defines the Markov decision process of generating a molecule.

    Adapted from: https://github.com/google-research/google-research/blob/master/mol_dqn/chemgraph/dqn/molecules.py"""

    def __init__(self, action_space: MoleculeActions = None, observation_space: Space = None,
                 reward: RewardFunction = None, init_mol=None, max_steps=10,
                 target_fn=None, record_path=False, fingerprint_size=2048, fingerprint_radius=3):
        """Initializes the parameters for the MDP.

        Internal state will be stored as SMILES strings, but but the environment will
        return the new state as an ML-ready fingerprint

        Args:
          init_mol: String, Chem.Mol, or Chem.RWMol. If string is provided, it is
            considered as the SMILES string. The molecule to be set as the initial
            state. If None, an empty molecule will be created.
          max_steps: Integer. The maximum number of steps to run.
          target_fn: A function or None. The function should have Args of a
            String, which is a SMILES string (the state), and Returns as
            a Boolean which indicates whether the input satisfies a criterion.
            If None, it will not be used as a criterion.
          record_path: Boolean. Whether to record the steps internally.
        """

        # Capture the user settings
        if action_space is None:
            action_space = MoleculeActions(['C', 'O', 'N', 'F'])
        if observation_space is None:
            observation_space = AllMolecules()
        if reward is None:
            reward = LogP()
        self._reward = reward
        self.action_space = action_space
        self.init_mol = init_mol
        self.max_steps = max_steps
        self.target_fn = target_fn
        self.record_path = record_path
        self.observation_space = observation_space

        # Store the function used to compute inputs
        self.fingerprint_function = partial(compute_morgan_fingerprints,
                                            fingerprint_length=fingerprint_size,
                                            fingerprint_radius=fingerprint_radius)

        # Define the state variables
        self._state = None
        self._state_fingerprint = None
        self._path = None
        self._counter = None

        # Ready the environment
        self.reset()

    @property
    def num_steps_taken(self):
        return self._counter

    @property
    def state(self):
        """State as a SMILES string"""
        return self._state

    def get_path(self):
        return list(self._path)

    def reset(self):
        """Resets the MDP to its initial state."""
        self._state = self.init_mol
        self._state_fingerprint = self.fingerprint_function(self._state)
        self.action_space.update_actions(self._state, self.observation_space)
        if self.record_path:
            self._path = [self._state]
        self._counter = 0

    def reward(self):
        """Gets the reward for the state.

        A child class can redefine the reward function if reward other than
        zero is desired.

        Returns:
          Float. The reward for the current state.
        """
        return self._reward(self._state)

    def step(self, action):
        """Takes a step forward according to the action.

        Args:
          action (ndarray): Fingerprint of action

        Raises:
          ValueError: If the number of steps taken exceeds the preset max_steps, or
            the action is not in the set of valid_actions.

        """
        if self._counter >= self.max_steps:
            raise ValueError('This episode is terminated.')

        # Get the SMILES string associated with this action
        self._state = self.action_space.get_smiles_from_fingerprint(action)
        if self.record_path:
            self._path.append(self._state)

        # Store the fingerprint of the state
        self._state_fingerprint = self.fingerprint_function(self._state)

        # Update the action space
        self.action_space.update_actions(self._state, self.observation_space)
        self._counter += 1

        # Check if we have finished
        #  Out of steps or no more moves
        done = ((self._counter >= self.max_steps) or
                len(self.action_space.get_possible_actions(smiles=True)) == 0)

        # Compute the fingerprints for the state
        return self._state_fingerprint, self.reward(), done, {}

    def render(self, mode='human', **kwargs):
        """Draws the molecule of the state.

        Args:
          **kwargs: The keyword arguments passed to Draw.MolToImage.

        Returns:
          A PIL image containing a drawing of the molecule.
        """
        return Draw.MolToImage(self._state, **kwargs)


