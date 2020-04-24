from molgym.envs.simple import Molecule


def test_reward():
    env = Molecule()
    assert env._state is None
    assert env.reward() == 0

    env = Molecule(init_mol='C')
    env.reward()
