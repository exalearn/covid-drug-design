import numpy as np
import pickle as pkl

from molgym.agents.moldqn import DQNFinalState
from molgym.agents.preprocessing import MorganFingerprints
from molgym.envs.simple import Molecule


def test_serialization():
    moldqn = DQNFinalState(Molecule(), MorganFingerprints(), batch_size=8)
    res = pkl.dumps(moldqn)
    moldqn_2 = pkl.loads(res)
    assert np.isclose(moldqn.action_network.get_weights()[0],
                      moldqn_2.action_network.get_weights()[0]).all()

    # Test after training
    moldqn.env.reset()
    for i in range(10):
        action, _, _ = moldqn.action()
        new_state, reward, done, _ = moldqn.env.step(action)
        # Save outcome
        moldqn.remember(moldqn.env.state, action, reward,
                        new_state, moldqn.env.action_space.get_possible_actions(), done)

    # Make a training step
    moldqn.train()

    # Test out the serialization
    res = pkl.dumps(moldqn)
    moldqn_2 = pkl.loads(res)
    assert len(moldqn_2.optimizer.get_weights()) == 0

