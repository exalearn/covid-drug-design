from math import isclose
import pickle as pkl

from molgym.envs.rewards.mpnn import MPNNReward

from molgym.utils.conversions import convert_smiles_to_nx


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

