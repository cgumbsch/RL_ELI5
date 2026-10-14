"""Every-visit Monte Carlo control time-lapse and comparison with value iteration.

Q is only updated at the end of each episode. Exploration epsilon_k = 1/k (GLIE) or 1/sqrt(k).

    python -m demos.mc_control [--episodes 2000] [--seed 678] [--epsilon 1/k|1/sqrt]
"""
import argparse

import algorithms as alg
import gridworld as gw
import render as rd
from demos.timelapse import SHOW_EPISODES, render_timelapse, report, save_comparison

EPSILON = {"1/k": lambda k: 1 / k, "1/sqrt": lambda k: k ** -0.5}


def main():
    p = rd.add_common_args(argparse.ArgumentParser(description=__doc__), "mc_control")
    p.add_argument("--seed", type=int, default=678)
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--epsilon", choices=sorted(EPSILON), default="1/k")
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--no-gif", action="store_true")
    a = p.parse_args()

    env = gw.GridWorld(a.layout)
    ff = [n for n in SHOW_EPISODES if 1 < n < a.episodes] + [a.episodes]
    res = alg.mc_control(env, a.episodes, epsilon=EPSILON[a.epsilon], seed=a.seed, max_steps=a.max_steps,
                         snapshot_at={1, *ff}, record_at={1, *ff}, record_steps_of={1})
    for f in render_timelapse(env, res, a.out, a.width, a.height, ff, gif=not a.no_gif):
        print("wrote", f)
    print("wrote", save_comparison(env, res, a.out, a.width, a.height))
    report(env, res)


if __name__ == "__main__":
    main()
