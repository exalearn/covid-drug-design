"""Different choices for actions to use in molecular design"""
from typing import List, Optional

import networkx as nx
import numpy as np
from gym import Space

from molgym.envs.actions.utils import rdkit, get_valid_actions


# TODO (wardlt): Extract a base class that defines the interface for `update_actions`
from molgym.utils.conversions import convert_nx_to_smiles, convert_smiles_to_nx


class MoleculeActions(Space):
    """Action space for molecule design

    Generates which molecules are possible next steps and stores
    them as potential actions, following the approach of
     `Zhou et al. <http://www.nature.com/articles/s41598-019-47148-x>`_."""

    def __init__(self, atom_types, allow_removal=True, allow_no_modification=False,
                 allow_bonds_between_rings=True, allowed_ring_sizes=None,
                 max_molecule_size: Optional[int] = None):
        """
        Args:
            atom_types: The set of elements the molecule may contain.
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
            max_molecule_size: int, Maximum number of heavy atoms
        """

        super().__init__((None, None), np.str)

        # Store the rules for defining actions
        self.atom_types = atom_types
        self.allow_removal = allow_removal
        self.allow_no_modification = allow_no_modification
        self.allow_bonds_between_rings = allow_bonds_between_rings
        self.allowed_ring_sizes = allowed_ring_sizes
        self._state = None
        self._valid_actions = []
        self._max_bonds = 4
        self.max_molecule_size = max_molecule_size
        atom_types = list(self.atom_types)
        self._max_new_bonds = dict(
            zip(atom_types, rdkit.atom_valences(atom_types))
        )

        # Placeholders for action space
        self._valid_actions = []

    def sample(self):
        return np.random.choice(self._valid_actions)

    def contains(self, x):
        return x in self._valid_actions

    @property
    def n(self):
        return len(self._valid_actions)

    def get_possible_actions(self) -> List[nx.Graph]:
        """Get the possible actions given the current state

        Returns:
            (list) List of the possible actions
        """
        return list(self._valid_actions)

    def update_actions(self, new_state: nx.Graph, allowed_space: Space):
        """Generate the available actions for a new state

        Uses the actions to redefine the action space for

        Args:
            new_state (str): Molecule used to define action space
            allowed_space (Space): Space of possible observations
        """

        # Store the new state
        self._state = new_state

        # Compute the possible actions, which we describe by the new molecule they would form
        valid_actions = get_valid_actions(
            convert_nx_to_smiles(new_state),
            atom_types=self.atom_types,
            allow_removal=self.allow_removal,
            allow_no_modification=self.allow_no_modification,
            allowed_ring_sizes=self.allowed_ring_sizes,
            allow_bonds_between_rings=self.allow_bonds_between_rings,
            max_molecule_size=self.max_molecule_size
        )

        # Get only those actions which are in the desired space
        self._valid_actions = [convert_smiles_to_nx(x) for x in valid_actions if x in allowed_space]
