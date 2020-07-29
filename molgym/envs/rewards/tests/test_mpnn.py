from math import isclose
import pickle as pkl
import json
import os

from tensorflow.keras.models import load_model
from pytest import fixture

from molgym.envs.rewards.mpnn import MPNNReward
from molgym.mpnn.layers import custom_objects
from molgym.utils.conversions import convert_smiles_to_nx

_home_dir = os.path.dirname(__file__)


@fixture
def model():
    return load_model(os.path.join(_home_dir, 'model.h5'),
                      custom_objects=custom_objects)


@fixture
def atom_types():
    with open(os.path.join(_home_dir, 'atom_types.json')) as fp:
        return json.load(fp)


@fixture
def bond_types():
    with open(os.path.join(_home_dir, 'bond_types.json')) as fp:
        return json.load(fp)


def test_mpnn_reward(model, atom_types, bond_types):
    reward = MPNNReward(model, atom_types, bond_types)
    graph = convert_smiles_to_nx('CCC')
    assert isinstance(reward(graph), float)

def test_pickle(model, atom_types, bond_types):
    # Run inference on the first graph
    reward = MPNNReward(model, atom_types, bond_types)
    graph = convert_smiles_to_nx('CCC')
    reward(graph)

    # Clone the model
    reward2 = pkl.loads(pkl.dumps(reward))

    assert isclose(reward(graph), reward2(graph), abs_tol=1e-6)

