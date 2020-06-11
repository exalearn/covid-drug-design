"""Test multiobjective rewards"""
from math import isclose

from molgym.envs.rewards import RewardFunction
from molgym.envs.rewards.multiobjective import AdditiveReward


class ExampleReward(RewardFunction):

    def _call(self, graph) -> float:
        return 2


def test_additive():
    r1 = ExampleReward(maximize=True)
    r2 = ExampleReward(maximize=False)
    mr = AdditiveReward([
        {'reward': r1},
        {'reward': r2, 'mean': 1, 'scale': 2}
    ])
    assert isclose(mr(None), 2 - (2 - 1) / 2)
