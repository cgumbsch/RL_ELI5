"""Q-learning time-lapse and comparison with value iteration.

epsilon = 1 for the first episodes, then decays linearly to --eps-min; alpha = 1 / N(s,a)^omega.

    python -m demos.q_learning [--episodes 2000] [--seed 72] [--explore 500,1000] [--eps-min 0.2]
"""
import argparse

import algorithms as alg
import gridworld as gw
import render as rd
from demos.timelapse import SHOW_EPISODES, render_timelapse, report, save_comparison


def epsilon_schedule(explore_until, reach_min_at, eps_min):
    def eps(n):
        if n <= explore_until:
            return 1.0
        if n >= reach_min_at:
            return eps_min
        return 1.0 - (1.0 - eps_min) * (n - explore_until) / (reach_min_at - explore_until)
    return eps


def main():
    p = rd.add_common_args(argparse.ArgumentParser(description=__doc__), "q_learning")
    p.add_argument("--seed", type=int, default=72)
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--omega", type=float, default=0.7)
    p.add_argument("--explore", default="500,1000", help="epsilon = 1 until A, then linear to --eps-min at B")
    p.add_argument("--eps-min", type=float, default=0.2)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--no-gif", action="store_true")
    a = p.parse_args()

    env = gw.GridWorld(a.layout)
    e1, e2 = (int(v) for v in a.explore.split(","))
    ff = [n for n in SHOW_EPISODES if 1 < n < a.episodes] + [a.episodes]
    res = alg.q_learning(env, a.episodes, epsilon=epsilon_schedule(e1, e2, a.eps_min), omega=a.omega,
                         seed=a.seed, max_steps=a.max_steps, snapshot_at={1, *ff}, record_at={1, *ff},
                         record_steps_of={1})
    for f in render_timelapse(env, res, a.out, a.width, a.height, ff, gif=not a.no_gif):
        print("wrote", f)
    print("wrote", save_comparison(env, res, a.out, a.width, a.height))
    report(env, res)


if __name__ == "__main__":
    main()
