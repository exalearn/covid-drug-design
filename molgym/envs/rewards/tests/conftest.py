import json
import os

from tensorflow.keras.models import load_model
from pytest import fixture

from molgym.mpnn.layers import custom_objects
from molgym.utils.conversions import convert_smiles_to_nx


_home_dir = os.path.dirname(__file__)


@fixture
def oneshot_model():
    return load_model(os.path.join(_home_dir, 'oneshot_model.h5'),
                      custom_objects=custom_objects)

@fixture
def target_mols():
    return [convert_smiles_to_nx(x) for x in ['CCC', 'CCCC']]


@fixture
def bond_types():
    with open(os.path.join(_home_dir, 'bond_types.json')) as fp:
        return json.load(fp)


@fixture
def model():
    return load_model(os.path.join(_home_dir, 'model.h5'),
                      custom_objects=custom_objects)


@fixture
def atom_types():
    with open(os.path.join(_home_dir, 'atom_types.json')) as fp:
        return json.load(fp)
