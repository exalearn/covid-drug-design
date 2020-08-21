from math import isclose
import pickle as pkl
import numpy as np
import json
import os

from tensorflow.keras.models import load_model
from pytest import fixture

from molgym.envs.rewards.oneshot import OneShotScore
from molgym.mpnn.layers import custom_objects
from molgym.utils.conversions import convert_smiles_to_nx

_home_dir = os.path.dirname(__file__)


@fixture
def model():
    return load_model(os.path.join(_home_dir, 'oneshot_model.h5'),
                      custom_objects=custom_objects)


@fixture
def atom_types():
    with open(os.path.join(_home_dir, 'atom_types.json')) as fp:
        return json.load(fp)


@fixture
def bond_types():
    with open(os.path.join(_home_dir, 'bond_types.json')) as fp:
        return json.load(fp)


@fixture
def target_mols():
    return [convert_smiles_to_nx(x) for x in ['CCC', 'CCCC']]


def test_reward(model, atom_types, bond_types, target_mols):
    reward = OneShotScore(model, atom_types, bond_types, target_mols)
    graph = convert_smiles_to_nx('CCC')
    assert isinstance(reward(graph), float)


def test_pickle(model, atom_types, bond_types, target_mols):
    # Run inference on the first graph
    reward = OneShotScore(model, atom_types, bond_types, target_mols)
    graph = convert_smiles_to_nx('CCC')
    reward(graph)

    # Clone the model
    reward2 = pkl.loads(pkl.dumps(reward))

    assert isclose(reward(graph), reward2(graph), abs_tol=1e-6)


