import numpy as np
import pytest

import algorithms as alg
import gridworld as gw

W, G = "wall", "goal"


@pytest.fixture(scope="module")
def env():
    return gw.GridWorld()


@pytest.fixture(scope="module")
def V(env):
    return alg.value_iteration(env)[-1]


def test_optimal_values(env, V):
    grid = [[W if (r, c) in env.walls else G if (r, c) == env.goal else round(V[(r, c)], 2)
             for c in range(env.cols)] for r in range(env.rows)]
    assert grid == [
        [W,    4.86, 5.40, 6.00, G],
        [4.30, 4.78, W,    9.00, 10.00],
        [4.78, 5.31, W,    8.10, 9.00],
        [5.31, 5.90, 6.56, 7.29, 8.10],
    ]


def test_puddle_backup(env, V):
    assert env.expected_reward(env.puddle, "left") == pytest.approx(-2.0)
    assert alg.q_from_v(env, V, env.puddle, "right") == pytest.approx(6.0)


def test_optimal_path_avoids_puddle(env, V):
    pi = alg.greedy_policy(env, V)
    assert alg.path(env, pi) == [(2, 0), (2, 1), (3, 1), (3, 2), (3, 3), (2, 3), (1, 3), (1, 4), (0, 4)]
    assert alg.policy_evaluation(env, pi)[env.start] == pytest.approx(V[env.start])


def test_slip_probability(env):
    rng = gw.make_rng(0)
    slips = [env.step(env.puddle, "right", rng)[3] for _ in range(20000)]
    assert np.mean(slips) == pytest.approx(env.slip_prob, abs=0.01)


def test_returns_to_go():
    assert alg.returns_to_go([0, 0, 10], 0.9) == pytest.approx([8.1, 9.0, 10.0])
    assert alg.discounted_return([0, 0, 10], 0.9) == pytest.approx(8.1)


@pytest.mark.parametrize("learn, kwargs", [
    (alg.q_learning, dict(epsilon=0.3, seed=0)),
    (alg.mc_control, dict(epsilon=0.3, seed=0)),
])
def test_learning_finds_safe_route(env, V, learn, kwargs):
    res = learn(env, 3000, **kwargs)
    pi = {s: {acts[0]: 1.0} for s, acts in alg.greedy_actions(env, res["Q"]).items()}
    path = alg.path(env, pi, max_len=30)
    assert path[-1] == env.goal and env.puddle not in path
