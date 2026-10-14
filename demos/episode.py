"""Animate one episode: the robot follows a policy and leaves a trace.

    python -m demos.episode [--policy optimal|random] [--actions "up up right"]
                            [--slip | --no-slip] [--seed 0]
"""
import argparse
from pathlib import Path

import numpy as np

import algorithms as alg
import gridworld as gw
import render as rd

BUMP_DIST = 0.18
SLIP_START_FRAMES = 3
SLIP_START_DIST = 0.1
FALL_FRAME_REPEAT = 2
FALL_LIFT = 0.15
LEAD_SECONDS = 0.6


def run_episode(env, policy, actions, slip, seed, max_steps):
    rng = gw.make_rng(seed)
    if not actions:
        pi = alg.greedy_policy(env, alg.value_iteration(env)[-1]) if policy == "optimal" else alg.random_policy(env)
        return alg.rollout(env, pi, rng, slip=slip, max_steps=max_steps)
    steps, s = [], env.start
    for a in actions:
        s2, r, done, slipped = env.step(s, a, rng, slip=slip)
        steps.append((s, a, r, s2, done, slipped))
        if done:
            break
        s = s2
    return steps


def animate(env, steps, out, width, height, gif=True):
    cells = [env.start] + [st[3] for st in steps]
    slipped = bool(steps) and steps[-1][5]
    cv = rd.Canvas(env, width, height)
    rd.draw_grid(cv)
    trace = rd.draw_trace(cv, [rd.center(env.start)])
    n_step = round(rd.STEP_SECONDS * rd.FPS)

    def frame(pos, sprite, visited, sheet="walk.png", trace_end=None):
        trace.set_data(*zip(*[rd.center(c) for c in visited + [trace_end or pos]]))
        art = rd.draw_robot(cv, pos, frame=sprite, sheet=sheet)
        img = cv.to_array()
        art.remove()
        return img

    def frames():
        for _ in range(round(LEAD_SECONDS * rd.FPS)):
            yield frame(env.start, rd.STAND_FRAME, [])
        t = 0
        for k in range(len(cells) - 1 - slipped):
            p0, p1 = np.array(cells[k], float), np.array(cells[k + 1], float)
            d = np.array(gw.DELTAS[steps[k][1]], float)
            for f in range(1, n_step + 1):
                if cells[k] == cells[k + 1]:
                    pos = tuple(p0 + d * BUMP_DIST * (1 - abs(2 * f / n_step - 1)))
                else:
                    pos = tuple(p0 + (p1 - p0) * f / n_step)
                yield frame(pos, rd.WALK_CYCLE[t % len(rd.WALK_CYCLE)], cells[:k + 1])
                t += 1
        if not slipped:
            for _ in range(round(rd.HOLD_SECONDS * rd.FPS)):
                yield frame(cells[-1], rd.STAND_FRAME, cells[:-1])
            return
        here = np.array(cells[-1], float)
        d = np.array(gw.DELTAS[steps[-1][1]], float)
        visited = cells[:-1]
        for f in range(1, SLIP_START_FRAMES + 1):
            pos = tuple(here + d * SLIP_START_DIST * f / SLIP_START_FRAMES)
            yield frame(pos, rd.WALK_CYCLE[t % len(rd.WALK_CYCLE)], visited, trace_end=cells[-1])
            t += 1
        off = rd.fall_offsets()
        drift_x = (pos[1] - here[1]) + off[rd.FALL_SEQUENCE[-1]]
        for j, k in enumerate(rd.FALL_SEQUENCE):
            ramp = (j + 1) / len(rd.FALL_SEQUENCE)
            fpos = (here[0] - FALL_LIFT * ramp, pos[1] + off[k] - drift_x * ramp)
            for _ in range(FALL_FRAME_REPEAT):
                yield frame(fpos, k, visited, sheet="fall.png", trace_end=cells[-1])
        for _ in range(round(rd.HOLD_SECONDS * rd.FPS)):
            yield frame(fpos, rd.FALL_SEQUENCE[-1], visited, sheet="fall.png", trace_end=cells[-1])

    files = rd.save_animation(frames(), out, gif=gif)
    cv.close()
    return files


def main():
    p = rd.add_common_args(argparse.ArgumentParser(description=__doc__), "episode")
    p.add_argument("--policy", choices=("optimal", "random"), default="optimal")
    p.add_argument("--actions", default="", help='fixed action sequence, e.g. "up up right"')
    g = p.add_mutually_exclusive_group()
    g.add_argument("--slip", dest="slip", action="store_true", help="always slip on the puddle")
    g.add_argument("--no-slip", dest="slip", action="store_false", help="never slip on the puddle")
    p.set_defaults(slip=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=60)
    p.add_argument("--no-gif", action="store_true")
    a = p.parse_args()

    env = gw.GridWorld(a.layout)
    steps = run_episode(env, a.policy, a.actions.split(), a.slip, a.seed, a.max_steps)
    for f in animate(env, steps, Path(a.out), a.width, a.height, gif=not a.no_gif):
        print("wrote", f)
    G = alg.discounted_return([st[2] for st in steps], env.gamma)
    print(f"{len(steps)} steps, G = {G:.2f}")


if __name__ == "__main__":
    main()
