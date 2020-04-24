from molgym.envs.simple import Molecule


def test_reward():
    env = Molecule(init_mol='C')
    env.reward()
