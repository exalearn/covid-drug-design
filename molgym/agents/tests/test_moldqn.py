import numpy as np
import pickle as pkl

from molgym.agents.moldqn import DQNFinalState
from molgym.agents.preprocessing import MorganFingerprints
from molgym.envs.simple import Molecule


def test_serialization():
    moldqn = DQNFinalState(Molecule(), MorganFingerprints())
    res = pkl.dumps(moldqn)
    moldqn_2 = pkl.loads(res)
    assert np.isclose(moldqn.action_network.get_weights()[0],
                      moldqn_2.action_network.get_weights()[0]).all()
