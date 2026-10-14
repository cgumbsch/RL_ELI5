"""Shared visuals for the learning demos: a time-lapse of max_a Q while the agent learns,
and a side-by-side of the learned values against V* from value iteration."""
import numpy as np
from PIL import Image

import algorithms as alg
import gridworld as gw
import render as rd

SHOW_EPISODES = (2, 5, 10, 20, 50, 100, 200, 500, 1000)
STEP_SECONDS_EP1 = 0.08
EP1_HOLD = 0.3
FF_RUN, FF_HOLD = 0.36, 0.24
ARROWS_FADE, ARROWS_HOLD = 0.6, 1.2
TRAIL_ALPHA = 0.55


class HeatGrid:
    def __init__(self, cv, colorbar_label=None):
        self.cells = [s for s in cv.env.states if not cv.env.is_terminal(s)]
        self.tiles = rd.draw_grid(cv, fill={s: rd.UNVISITED for s in self.cells})
        if colorbar_label:
            rd.draw_colorbar(cv, colorbar_label)
        self.labels = rd.value_labels(cv)

    def set(self, values, visited, prev=None, blend=1.0):
        """Unvisited cells stay grey; `prev` and `blend` fade the colours from an earlier snapshot."""
        for s in self.cells:
            if s not in visited:
                self.tiles[s].set_facecolor(rd.UNVISITED)
                self.labels[s].set_visible(False)
                continue
            v = values[s]
            if prev is not None and s in prev[1]:
                v = prev[0][s] + (v - prev[0][s]) * blend
            self.tiles[s].set_facecolor(rd.value_color(v))
            self.labels[s].set_text(f"{values[s]:.2f}")
            self.labels[s].set_visible(True)


def episode_counter(cv):
    in_wall = (0, 0) in cv.env.walls
    cx, cy = (0.5, 0.5) if in_wall else (0.5, -0.45)
    col = "white" if in_wall else rd.INK
    cv.ax.text(cx, cy - 0.17, "episode", ha="center", va="center", color=col, fontsize=rd.pt(cv, 30), zorder=20)
    return cv.ax.text(cx, cy + 0.12, "1", ha="center", va="center", fontweight="bold", color=col,
                      fontsize=rd.pt(cv, 46), zorder=20)


def render_timelapse(env, res, out, width, height, ff, gif=True):
    """Episode 1 in full (Q after each step from res["step_snapshots"][1]),
    then fast-forward through the episodes in `ff`, then the greedy arrows fade in."""
    fps = rd.FPS
    cv = rd.Canvas(env, width, height, box=(0.02, 0.06, 0.86, 0.94))
    hg = HeatGrid(cv, colorbar_label=r"$\max_a Q(s,a)$")
    count = episode_counter(cv)
    trail = rd.draw_trace(cv, [], alpha=TRAIL_ALPHA)
    walk_t = [0]

    def robot(pos, lying=False):
        if lying:
            return rd.draw_fallen_robot(cv, pos)
        walk_t[0] += 1
        return rd.draw_robot(cv, pos, frame=rd.WALK_CYCLE[walk_t[0] % len(rd.WALK_CYCLE)])

    def standing(pos, lying):
        return rd.draw_fallen_robot(cv, pos) if lying else rd.draw_robot(cv, pos)

    def grab(art):
        img = cv.to_array()
        art.remove()
        return img

    def at(steps, u):
        """Robot position after u (fractional) steps, and the number of completed steps."""
        k = min(int(u), len(steps) - 1)
        f = 1.0 if u >= len(steps) else u - k
        s, act, _, s2, _, _ = steps[k]
        p0 = np.array(s, float)
        if s == s2:
            d = np.array(gw.DELTAS[act], float)
            return tuple(p0 + d * 0.18 * (1 - abs(2 * f - 1))), min(len(steps), int(u))
        return tuple(p0 + (np.array(s2, float) - p0) * f), min(len(steps), int(u))

    def show_trail(cells, pos=None):
        pts = [rd.center(c) for c in cells] + ([rd.center(pos)] if pos is not None else [])
        trail.set_data(*zip(*pts))

    def first_episode():
        steps, snaps = res["episodes"][1], res["step_snapshots"][1]
        cells = [steps[0][0]] + [st[3] for st in steps]
        zero = {s: 0.0 for s in hg.cells}
        F = max(1, round(len(steps) * STEP_SECONDS_EP1 * fps))
        for i in range(1, F + 1):
            pos, k = at(steps, i / F * len(steps))
            if k == len(steps):
                Qk, vis = res["snapshots"][1]
            else:
                Qk, vis = snaps[k - 1] if k >= 1 else (None, {env.start})
            hg.set(alg.max_q(env, Qk) if Qk is not None else zero, vis)
            show_trail(cells[:k + 1], pos)
            yield grab(robot(pos, lying=(i == F and steps[-1][5])))
        for _ in range(round(EP1_HOLD * fps)):
            yield grab(standing(cells[-1], steps[-1][5]))
        Q1, vis1 = res["snapshots"][1]
        return (alg.max_q(env, Q1), vis1), cells[-1], steps[-1][5]

    def fast_forward(n, prev):
        Qn, vis = res["snapshots"][n]
        vals = alg.max_q(env, Qn)
        steps = res["episodes"][n]
        cells = [steps[0][0]] + [st[3] for st in steps]
        count.set_text(f"{n}")
        n_run = max(1, round(FF_RUN * fps))
        for i in range(1, n_run + 1):
            pos, k = at(steps, i / n_run * len(steps))
            show_trail(cells[:k + 1], pos)
            hg.set(vals, vis | prev[1], prev=prev, blend=i / n_run)
            yield grab(robot(pos, lying=(i == n_run and steps[-1][5])))
        show_trail(cells)
        hg.set(vals, vis)
        for _ in range(round(FF_HOLD * fps)):
            yield grab(standing(cells[-1], steps[-1][5]))
        return (vals, vis), cells[-1], steps[-1][5]

    def frames():
        prev, end, lying = yield from first_episode()
        for n in ff:
            prev, end, lying = yield from fast_forward(n, prev)
        arrows = rd.draw_policy_arrows(cv, alg.greedy_actions(env, res["Q"]))
        nf = round(ARROWS_FADE * fps)
        for i in range(nf):
            w = (i + 1) / nf
            for ar in arrows:
                ar.set_alpha(w)
            trail.set_alpha(TRAIL_ALPHA * (1 - w))
            yield grab(standing(end, lying))
        for _ in range(round(ARROWS_HOLD * fps)):
            yield grab(standing(end, lying))

    files = rd.save_animation(frames(), out, gif=gif)
    cv.close()
    return files


def panel(env, values, actions, title, width, height, visited=None):
    cv = rd.Canvas(env, width, height, box=(0.05, 0.12, 0.95, 0.97))
    HeatGrid(cv).set(values, visited if visited is not None else set(env.states))
    rd.draw_policy_arrows(cv, actions)
    top_px = cv.ax.transData.transform((0, -cv.border))[1]
    fs_px = 40 * min(1.0, cv.px(1) / 160)
    cv.fig.text(0.5, (top_px + 0.9 * fs_px) / height, title, ha="center", va="bottom", color=rd.INK,
                fontsize=fs_px * 72 / rd.DPI, fontweight="bold")
    img = cv.to_array()
    cv.close()
    return img


def save_comparison(env, res, out, width, height):
    """Learned max_a Q next to V* from value iteration."""
    V = alg.value_iteration(env)[-1]
    Q = res["Q"]
    W = width // 2
    img = np.concatenate([
        panel(env, alg.max_q(env, Q), alg.greedy_actions(env, Q), "learned from experience",
              W, height, visited=res["visited"]),
        panel(env, V, alg.optimal_actions(env, V), "value iteration (model)", W, height),
    ], axis=1)
    path = f"{out}_compare.png"
    Image.fromarray(img).save(path)
    return path


def report(env, res):
    V = alg.value_iteration(env)[-1]
    M = alg.max_q(env, res["Q"])
    worst = max(M, key=lambda s: abs(M[s] - V[s]))
    pi = {s: {acts[0]: 1.0} for s, acts in alg.greedy_actions(env, res["Q"]).items()}
    print(f"max |max_a Q - V*| = {abs(M[worst] - V[worst]):.3f} at {worst}")
    print(f"greedy path: {alg.path(env, pi, max_len=30)}")
