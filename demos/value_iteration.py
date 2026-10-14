"""Optimal values V* from value iteration as a heatmap with the greedy policy.

    python -m demos.value_iteration [--sweeps K] [--no-values] [--no-arrows]
"""
import argparse

import algorithms as alg
import gridworld as gw
import render as rd


def main():
    p = rd.add_common_args(argparse.ArgumentParser(description=__doc__), "value_iteration")
    p.add_argument("--sweeps", type=int, default=None, help="stop after K sweeps (default: converge)")
    p.add_argument("--no-values", action="store_true")
    p.add_argument("--no-arrows", action="store_true")
    a = p.parse_args()

    env = gw.GridWorld(a.layout)
    history = alg.value_iteration(env)
    V = history[min(a.sweeps, len(history) - 1)] if a.sweeps is not None else history[-1]
    print(f"converged after {len(history) - 1} sweeps, V(start) = {history[-1][env.start]:.2f}")

    cv = rd.Canvas(env, a.width, a.height, box=(0.02, 0.06, 0.86, 0.94))
    rd.draw_grid(cv, fill={s: rd.value_color(V[s]) for s in env.states if not env.is_terminal(s)})
    rd.draw_colorbar(cv, r"$V(s)$")
    if not a.no_values:
        rd.draw_values(cv, V, dy=0.0 if a.no_arrows else -0.14)
    if not a.no_arrows:
        rd.draw_policy_arrows(cv, alg.optimal_actions(env, V), center_dy=0.0 if a.no_values else 0.25)
    print("wrote", cv.save(f"{a.out}.png"))
    cv.close()


if __name__ == "__main__":
    main()
