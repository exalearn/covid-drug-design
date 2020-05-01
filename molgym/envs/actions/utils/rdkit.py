# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python2, python3
"""Tools for manipulating graphs and converting from atom and pair features.

Adapted from: https://github.com/google-research/google-research/blob/master/mol_dqn/chemgraph/dqn/py/molecules.py
"""

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def atom_valences(atom_types):
    """Creates a list of valences corresponding to atom_types.

    Note that this is not a count of valence electrons, but a count of the
    maximum number of bonds each element will make. For example, passing
    atom_types ['C', 'H', 'O'] will return [4, 1, 2].

    Args:
      atom_types: List of string atom types, e.g. ['C', 'H', 'O'].

    Returns:
      List of integer atom valences.
    """
    periodic_table = Chem.GetPeriodicTable()
    return [
        max(list(periodic_table.GetValenceList(atom_type)))
        for atom_type in atom_types
    ]


def get_scaffold(mol):
    """Computes the Bemis-Murcko scaffold for a molecule.

    Args:
      mol: RDKit Mol.

    Returns:
      String scaffold SMILES.
    """
    return Chem.MolToSmiles(
        MurckoScaffold.GetScaffoldForMol(mol), isomericSmiles=True)


def contains_scaffold(mol, scaffold):
    """Returns whether mol contains the given scaffold.

    NOTE: This is more advanced than simply computing scaffold equality (i.e.
    scaffold(mol_a) == scaffold(mol_b)). This method allows the target scaffold to
    be a subset of the (possibly larger) scaffold in mol.

    Args:
      mol: RDKit Mol.
      scaffold: String scaffold SMILES.

    Returns:
      Boolean whether scaffold is found in mol.
    """
    pattern = Chem.MolFromSmiles(scaffold)
    matches = mol.GetSubstructMatches(pattern)
    return bool(matches)


def get_largest_ring_size(molecule):
    """Calculates the largest ring size in the molecule.

    Refactored from
    https://github.com/wengong-jin/icml18-jtnn/blob/master/bo/run_bo.py

    Args:
      molecule: Chem.Mol. A molecule.

    Returns:
      Integer. The largest ring size.
    """
    cycle_list = molecule.GetRingInfo().AtomRings()
    if cycle_list:
        cycle_length = max([len(j) for j in cycle_list])
    else:
        cycle_length = 0
    return cycle_length
