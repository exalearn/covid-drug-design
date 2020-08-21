from math import isclose
import pickle as pkl

from molgym.envs.rewards.oneshot import OneShotScore
from molgym.utils.conversions import convert_smiles_to_nx


def test_reward(oneshot_model, atom_types, bond_types, target_mols):
    reward = OneShotScore(oneshot_model, atom_types, bond_types, target_mols)
    graph = convert_smiles_to_nx('CCC')
    assert isinstance(reward(graph), float)


def test_pickle(oneshot_model, atom_types, bond_types, target_mols):
    # Run inference on the first graph
    reward = OneShotScore(oneshot_model, atom_types, bond_types, target_mols)
    graph = convert_smiles_to_nx('CCC')
    reward(graph)

    # Clone the model
    reward2 = pkl.loads(pkl.dumps(reward))

    assert isclose(reward(graph), reward2(graph), abs_tol=1e-6)


