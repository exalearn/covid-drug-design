"""Different choices for actions to use in molecular design"""

import copy
from functools import partial

import numpy as np
from gym import Space

from molgym.envs.utils import rdkit, get_valid_actions, _compute_canonical_smiles, compute_morgan_fingerprints


class MoleculeActions(Space):
    """Action space for molecule design

    Generates which molecules are possible next steps and stores
    them as potential actions, following the approach of
     `Zhou et al. <http://www.nature.com/articles/s41598-019-47148-x>`_."""

    def __init__(self, atom_types, allow_removal=True, allow_no_modification=False,
                 allow_bonds_between_rings=True, allowed_ring_sizes=None,
                 fingerprint_size=2048, fingerprint_radius=3):
        """
        Args:
            atom_types: The set of elements the molecule may contain.
            state. If None, an empty molecule will be created.
            allow_removal: Boolean. Whether to allow removal of a bond.
            allow_no_modification: Boolean. If true, the valid action set will
                include doing nothing to the current molecule, i.e., the current
                molecule itself will be added to the action set.
            allow_bonds_between_rings: Boolean. If False, new bonds connecting two
                atoms which are both in rings are not allowed.
                DANGER Set this to False will disable some of the transformations eg.
                c2ccc(Cc1ccccc1)cc2 -> c1ccc3c(c1)Cc2ccccc23
                But it will make the molecules generated make more sense chemically.
            allowed_ring_sizes: Set of integers or None. The size of the ring which
                is allowed to form. If None, all sizes will be allowed. If a set is
                provided, only sizes in the set is allowed.
             fingerprint_size (int): Length of the fingerprint used to represent each molecule
             fingerprint_radius (int): Size of the radius to include for the
        """

        super().__init__((None, fingerprint_size), np.int)

        # Store the rules for defining actions
        self.atom_types = atom_types
        self.allow_removal = allow_removal
        self.allow_no_modification = allow_no_modification
        self.allow_bonds_between_rings = allow_bonds_between_rings
        self.allowed_ring_sizes = allowed_ring_sizes
        self._state = None
        self._valid_actions = []
        self._max_bonds = 4
        atom_types = list(self.atom_types)
        self._max_new_bonds = dict(
            list(zip(atom_types, rdkit.atom_valences(atom_types)))
        )

        # Store the function for computing features
        self.fingerprint_function = partial(compute_morgan_fingerprints,
                                            fingerprint_length=fingerprint_size,
                                            fingerprint_radius=fingerprint_radius)

        # Placeholders for action space
        self._valid_actions = []
        self._valid_actions_featurized = []

    def sample(self):
        return self.np_random.randint(0, len(self._valid_actions))

    def contains(self, x):
        return x in self._valid_actions_featurized

    @property
    def n(self):
        return len(self._valid_actions)

    def get_possible_actions(self, smiles=False):
        """Get the possible actions given the current state

        Args:
            smiles (bool): Whether to return the smiles strings, or the featurized molecules
        Returns:
            (ndarray) List of the possible actions
        """
        output = self._valid_actions if smiles else self._valid_actions_featurized
        return copy.deepcopy(output)

    def update_actions(self, new_state, allowed_space: Space):
        """Generate the available actions for a new state

        Uses the actions to redefine the action space for

        Args:
            new_state (str): Molecule used to define action space
            allowed_space (Space): Space of possible observations
        """

        # Store the new state
        self._state = new_state

        # Compute the possible actions, which we describe by the new molecule they would form
        self._valid_actions = get_valid_actions(
            new_state,
            atom_types=self.atom_types,
            allow_removal=self.allow_removal,
            allow_no_modification=self.allow_no_modification,
            allowed_ring_sizes=self.allowed_ring_sizes,
            allow_bonds_between_rings=self.allow_bonds_between_rings)

        # Get only those actions which are in the desired space
        self._valid_actions = np.array([x for x in self._valid_actions
                                        if _compute_canonical_smiles(x) in allowed_space])

        # Compute the features for the next states
        self._valid_actions_featurized = np.array([self.fingerprint_function(m)
                                                   for m in self._valid_actions])

    def get_smiles_from_fingerprint(self, action):
        """Lookup the smiles string for an action given its fingerprint

        Args:
            action (ndarray): Fingerprint of a certain action
        Returns:
            (str) SMILES string associated with that action
        """

        for fingerprint, smiles in zip(self._valid_actions_featurized, self._valid_actions):
            if np.array_equal(fingerprint, action):
                return smiles
        raise ValueError('Action not found in current action space')
