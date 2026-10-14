"""Many random-policy episodes from the start, coloured by the return from each step on:
orange = 0, green = positive, red = negative.

    python -m demos.random_rollouts [--n 50] [--steps 20] [--seconds 4]
"""
import argparse

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb

import algorithms as alg
import gridworld as gw
import render as rd

GROW_SECONDS = 0.6
WAVE_STEP_SECONDS = 0.08
COLORS = {"zero": rd.TRACE_COLOR, "pos": "#2E9E4F", "neg": "#D62728"}
MAX_ALPHA = 0.85


def segments(steps, offset, gamma):
    """[(p0, p1, G_t)] for every step that moves."""
    returns = alg.returns_to_go([st[2] for st in steps], gamma)
    return [((s[1] + 0.5 + offset[0], s[0] + 0.5 + offset[1]),
             (s2[1] + 0.5 + offset[0], s2[0] + 0.5 + offset[1]), g)
            for (s, _, _, s2, _, _), g in zip(steps, returns) if s != s2]


def value_rgba(g, base_alpha, scale=10.0):
    w = min(1.0, abs(g) / scale)
    c0 = np.array(to_rgb(COLORS["zero"]))
    c1 = np.array(to_rgb(COLORS["pos" if g > 0 else "neg"]))
    return np.array((*(c0 + (c1 - c0) * w), base_alpha + (MAX_ALPHA - base_alpha) * w))


def partial(segs, u):
    n = len(segs)
    if n == 0 or u >= 1:
        return segs
    i, frac = int(u * n), u * n - int(u * n)
    out = list(segs[:i])
    p0, p1, g = segs[i]
    out.append((p0, (p0[0] + (p1[0] - p0[0]) * frac, p0[1] + (p1[1] - p0[1]) * frac), g))
    return out


def main():
    p = rd.add_common_args(argparse.ArgumentParser(description=__doc__), "random_rollouts")
    p.add_argument("--n", type=int, default=50, help="number of episodes")
    p.add_argument("--steps", type=int, default=20, help="maximum steps per episode")
    p.add_argument("--seconds", type=float, default=4.0)
    p.add_argument("--jitter", type=float, default=0.15, help="sideways offset per episode (cells)")
    p.add_argument("--alpha", type=float, default=0.25)
    p.add_argument("--no-gif", action="store_true")
    a = p.parse_args()

    env = gw.GridWorld(a.layout)
    pi = alg.random_policy(env)
    offsets = np.random.default_rng(12345)
    runs = []
    for seed in range(a.n):
        steps = alg.rollout(env, pi, gw.make_rng(seed), max_steps=a.steps)
        runs.append(segments(steps, offsets.uniform(-a.jitter, a.jitter, size=2), env.gamma))

    cv = rd.Canvas(env, a.width, a.height)
    rd.draw_grid(cv)
    lw = cv.px(0.03) * 72 / rd.DPI
    lines = [cv.ax.add_collection(LineCollection([], linewidths=lw, capstyle="round", joinstyle="round",
                                                 zorder=7 if any(g for *_, g in segs) else 6))
             for segs in runs]
    rd.draw_robot(cv, env.start)

    fps = rd.FPS
    stagger = max(0.0, a.seconds - GROW_SECONDS) / max(1, a.n - 1)
    longest = max(len(segs) for segs in runs)
    orange = value_rgba(0.0, a.alpha)

    def colours(segs, t_since_end):
        n, dt = len(segs), WAVE_STEP_SECONDS
        return [orange + (value_rgba(g, a.alpha) - orange) * np.clip((t_since_end - (n - 1 - j) * dt) / dt, 0, 1)
                for j, (*_, g) in enumerate(segs)]

    def frames():
        for f in range(round((a.seconds + longest * WAVE_STEP_SECONDS) * fps) + 1):
            t = f / fps
            for i, (line, segs) in enumerate(zip(lines, runs)):
                u = (t - i * stagger) / GROW_SECONDS
                if u > 0:
                    q = partial(segs, min(1.0, u))
                    line.set_segments([[p0, p1] for p0, p1, _ in q])
                    line.set_color([orange] * len(q) if u < 1 else colours(q, (u - 1) * GROW_SECONDS))
            yield cv.to_array()
        for _ in range(round(rd.HOLD_SECONDS * fps)):
            yield cv.to_array()

    for f in rd.save_animation(frames(), a.out, gif=not a.no_gif):
        print("wrote", f)
    cv.close()


if __name__ == "__main__":
    main()
