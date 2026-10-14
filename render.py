"""Drawing helpers: canvas, grid, sprites, values, arrows, video export.

Data coordinates: x = column, y = row (inverted), cell (r, c) spans [c, c+1] x [r, r+1].
"""
import argparse
import subprocess
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

import gridworld as gw

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
OUT = ROOT / "out"
DPI = 100
REF_CELL_PX = 214

BG = "#FFFFFF"
FLOOR = "#ECEBE5"
GRID_LINE = "#D2D1CA"
WALL = "#4B5157"
INK = "#141B1F"
ARROW_COLOR = "#5A6168"
TRACE_COLOR = "#E69F00"
UNVISITED = "#C9C9C6"

ROBOT_HEIGHT = 0.92
ROBOT_FOOT = 0.04
TILE_SCALE = 0.92
VALUE_FONT_PX = 44

FPS = 25
STEP_SECONDS = 0.4
HOLD_SECONDS = 2.0


def add_common_args(p: argparse.ArgumentParser, default_name: str):
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--layout", choices=sorted(gw.LAYOUTS), default="small")
    p.add_argument("--out", default=str(OUT / default_name), help="output path without extension")
    return p


def pt(cv, px_at_ref):
    """Size tuned at REF_CELL_PX cells -> points on this canvas."""
    return px_at_ref * cv.px(1) / REF_CELL_PX * 72 / DPI


def center(s):
    return s[1] + 0.5, s[0] + 0.5


# ---------------------------------------------------------------- canvas
class Canvas:
    """width x height px figure with the grid centred inside `box` (left, top, right, bottom fractions)."""

    def __init__(self, env, width=1920, height=1080, box=(0.06, 0.06, 0.94, 0.94), border=0.22, bg=BG):
        self.env, self.bg = env, bg
        self.width, self.height, self.border = width, height, border
        self.fig = plt.figure(figsize=(width / DPI, height / DPI), dpi=DPI, facecolor=bg)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_axis_off()
        self.ax.set_facecolor(bg)
        l, t, r, b = box
        self.cell_px = min(width * (r - l) / (env.cols + 2 * border), height * (b - t) / (env.rows + 2 * border))
        x0 = env.cols / 2 - width * (l + r) / 2 / self.cell_px
        y0 = env.rows / 2 - height * (t + b) / 2 / self.cell_px
        self.ax.set_xlim(x0, x0 + width / self.cell_px)
        self.ax.set_ylim(y0 + height / self.cell_px, y0)

    def px(self, cells):
        return cells * self.cell_px

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(path, dpi=DPI, facecolor=self.bg)
        return path

    def to_array(self):
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[..., :3].copy()

    def close(self):
        plt.close(self.fig)


# ---------------------------------------------------------------- images
@lru_cache(maxsize=None)
def load_asset(filename):
    return Image.open(ASSETS / filename).convert("RGBA")


@lru_cache(maxsize=None)
def sprite_frames(filename, min_width=20):
    """Split a one-row sprite sheet at transparent column gaps; frames share one vertical crop."""
    a = np.array(load_asset(filename))
    a[a[..., 3] < 10] = 0
    sheet = Image.fromarray(a)
    alpha = a[..., 3]
    segs, start = [], None
    for x, on in enumerate(np.append((alpha > 0).any(axis=0), False)):
        if on and start is None:
            start = x
        elif not on and start is not None:
            if x - start >= min_width:
                segs.append((start, x))
            start = None
    ys = np.where((alpha > 0).any(axis=1))[0]
    top, bottom = ys.min(), ys.max() + 1
    w = max(e - s for s, e in segs)
    frames = []
    for s, e in segs:
        f = Image.new("RGBA", (w, bottom - top), (0, 0, 0, 0))
        f.paste(sheet.crop((s, top, e, bottom)), ((w - (e - s)) // 2, 0))
        frames.append(f)
    return tuple(frames)


_RESIZE_CACHE = {}


def _resized(img, w, h):
    key = (id(img), w, h)
    if key not in _RESIZE_CACHE:
        _RESIZE_CACHE[key] = (img, np.asarray(img.resize((w, h), Image.LANCZOS)))
    return _RESIZE_CACHE[key][1]


def draw_image(cv, img, x, y, height, anchor="center", zorder=5):
    w = height * img.width / img.height
    arr = _resized(img, max(1, round(cv.px(w))), max(1, round(cv.px(height))))
    y_top = y - height / 2 if anchor == "center" else y - height
    return cv.ax.imshow(arr, extent=(x - w / 2, x + w / 2, y_top + height, y_top),
                        interpolation="nearest", zorder=zorder)


# ---------------------------------------------------------------- grid
def draw_grid(cv, fill=None):
    """fill: optional {state: colour}. Returns {state: floor Rectangle}."""
    fill = fill or {}
    ax, b, env = cv.ax, cv.border, cv.env
    ax.add_patch(FancyBboxPatch((-b, -b), env.cols + 2 * b, env.rows + 2 * b,
                                boxstyle=f"round,pad=0,rounding_size={b * 0.6}",
                                facecolor=WALL, edgecolor="none", zorder=0))
    lw = max(1.0, cv.px(0.012)) * 72 / DPI
    tiles = {s: ax.add_patch(Rectangle((s[1], s[0]), 1, 1, facecolor=fill.get(s, FLOOR),
                                       edgecolor=GRID_LINE, linewidth=lw, zorder=1))
             for s in env.states}
    draw_image(cv, load_asset("puddle_drops.png"), *center(env.puddle), TILE_SCALE, zorder=3)
    draw_image(cv, load_asset("goal_flag.png"), *center(env.goal), TILE_SCALE, zorder=3)
    return tiles


# ---------------------------------------------------------------- robot
STAND_FRAME = 0
WALK_CYCLE = tuple(range(1, 11))
FALL_SEQUENCE = tuple(range(1, 9))

# walk.png frames are hand-drawn and their upper bodies jitter slightly. walk_frames() keeps the
# upper body of one reference frame and only animates the legs, cut along the belt line.
_WALK_REF = 6
_BELT = dict(cx=70, y=184, curve=0.004)
_BODY_HALF, _LEGS_HALF, _HANDS_Y = 33, 37, 214


def _torso_offset(img, ref, rows=slice(125, 185), max_shift=6):
    best = None
    for dx in range(-max_shift, max_shift + 1):
        for dy in range(-max_shift, max_shift + 1):
            d = np.abs(np.roll(np.roll(img, dy, 0), dx, 1)[rows] - ref[rows]).mean()
            if best is None or d < best[0]:
                best = (d, dx, dy)
    return best[1], best[2]


@lru_cache(maxsize=None)
def walk_frames():
    A = [np.array(f) for f in sprite_frames("walk.png")]
    ref = A[_WALK_REF]
    h, w = ref.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    belt = _BELT["y"] - _BELT["curve"] * (xx - _BELT["cx"]) ** 2
    dist = np.abs(xx - _BELT["cx"])
    body = np.where(dist <= _BODY_HALF, yy < belt, yy < _HANDS_Y)
    legs = np.where(dist <= _BODY_HALF, yy >= belt - 2,
                    np.where(dist <= _LEGS_HALF, yy >= 207, yy >= _HANDS_Y))
    top = ref.copy()
    top[~body] = 0
    top = Image.fromarray(top)
    out = []
    for img in A:
        dx, dy = _torso_offset(img.astype(float), ref.astype(float))
        lower = np.roll(np.roll(img, dy, 0), dx, 1)
        lower[~legs] = 0
        f = Image.fromarray(lower)
        f.alpha_composite(top)
        out.append(f)
    return tuple(out)


@lru_cache(maxsize=None)
def _sheet_height(sheet):
    """Drawn height of a sheet's frames so every sheet shows the robot at the same size."""
    if sheet == "walk.png":
        return ROBOT_HEIGHT

    def standing_px(frames):
        ys = np.nonzero(np.array(frames[0])[..., 3] > 128)[0]
        return ys.max() - ys.min() + 1

    walk, other = sprite_frames("walk.png"), sprite_frames(sheet)
    return other[0].height * ROBOT_HEIGHT / walk[0].height * standing_px(walk) / standing_px(other)


@lru_cache(maxsize=None)
def fall_offsets(max_shift=60):
    """Horizontal shift per fall frame (cells) so consecutive silhouettes overlap best."""
    frames = sprite_frames("fall.png")
    masks = [np.array(f)[..., 3] > 128 for f in frames]
    px_to_cell = _sheet_height("fall.png") / frames[0].height
    cum = [0]
    for a, b in zip(masks, masks[1:]):
        ious = [((a & np.roll(b, dx, 1)).sum() / (a | np.roll(b, dx, 1)).sum(), dx)
                for dx in range(-max_shift, max_shift + 1)]
        cum.append(cum[-1] + max(ious)[1])
    base = cum[FALL_SEQUENCE[0]]
    return {k: (cum[k] - base) * px_to_cell for k in range(len(frames))}


def draw_robot(cv, s, frame=STAND_FRAME, sheet="walk.png", zorder=10):
    """Robot standing in cell s (fractional cells allowed). Returns the artist."""
    frames = walk_frames() if sheet == "walk.png" else sprite_frames(sheet)
    x, y = s[1] + 0.5, s[0] + 1 - ROBOT_FOOT
    return draw_image(cv, frames[frame % len(frames)], x, y, _sheet_height(sheet),
                      anchor="bottom", zorder=zorder)


def draw_fallen_robot(cv, s):
    return draw_robot(cv, (s[0] - 0.15, s[1]), frame=FALL_SEQUENCE[-1], sheet="fall.png")


# ---------------------------------------------------------------- values and arrows
VALUE_RANGE = (0.0, 10.0)
VALUE_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "values", matplotlib.colormaps["Greens"](np.linspace(0.04, 0.62, 256)))
VALUE_NORM = matplotlib.colors.Normalize(*VALUE_RANGE)


def value_color(v):
    return VALUE_CMAP(VALUE_NORM(v))


def value_labels(cv, dy=-0.14):
    """One hidden text artist per non-terminal state; fill in with .set_text()."""
    env = cv.env
    labels = {}
    for s in env.states:
        if env.is_terminal(s):
            continue
        halo = [pe.withStroke(linewidth=pt(cv, 8 if s == env.puddle else 5), foreground="white")]
        x, y = center(s)
        labels[s] = cv.ax.text(x, y + dy, "", ha="center", va="center", fontsize=pt(cv, VALUE_FONT_PX),
                               fontweight="bold", color=INK, zorder=20, path_effects=halo)
    return labels


def draw_values(cv, V, dy=-0.14):
    for s, label in value_labels(cv, dy).items():
        label.set_text(f"{V[s]:.2f}")


_DIR = {"up": (0, -1), "right": (1, 0), "down": (0, 1), "left": (-1, 0)}


def draw_policy_arrows(cv, actions, center_dy=0.25, length=0.26, lw_px=6, zorder=20):
    """actions: {state: [action, ...]}. Ties share one origin and read as a fork."""
    lw = pt(cv, lw_px)
    arts = []
    for s, acts in actions.items():
        if not acts:
            continue
        cx, cy = s[1] + 0.5, s[0] + 0.5 + center_dy
        if len(acts) == 1:
            mx, my = _DIR[acts[0]]
        else:
            mx = sum(_DIR[a][0] for a in acts) / len(acts)
            my = sum(_DIR[a][1] for a in acts) / len(acts)
        ox, oy = cx - mx * length / 2, cy - my * length / 2
        col = "white" if s == cv.env.puddle else ARROW_COLOR
        for a in acts:
            dx, dy = _DIR[a]
            arts.append(cv.ax.add_patch(FancyArrowPatch(
                (ox, oy), (ox + dx * length, oy + dy * length), arrowstyle="-|>",
                mutation_scale=lw * 4.5, linewidth=lw, color=col,
                shrinkA=0, shrinkB=0, capstyle="round", zorder=zorder)))
    return arts


def draw_colorbar(cv, label, gap=0.45, width=0.2, fontsize_px=38):
    env, b = cv.env, cv.border
    (x_px, top), (_, bottom) = cv.ax.transData.transform([(env.cols + b + gap, 0), (0, env.rows)])
    cax = cv.fig.add_axes([x_px / cv.width, bottom / cv.height,
                           cv.px(width) / cv.width, (top - bottom) / cv.height])
    cb = cv.fig.colorbar(matplotlib.cm.ScalarMappable(VALUE_NORM, VALUE_CMAP), cax=cax)
    cb.outline.set_visible(False)
    cb.set_ticks(np.arange(VALUE_RANGE[0], VALUE_RANGE[1] + 1e-9, 2.0))
    cax.tick_params(labelsize=pt(cv, fontsize_px), length=0, pad=12, colors=INK)
    cb.set_label(label, fontsize=pt(cv, fontsize_px), color=INK, labelpad=20)
    return cb


def draw_trace(cv, points, width=0.07, color=TRACE_COLOR, alpha=0.9, zorder=6):
    xs, ys = zip(*points) if points else ([], [])
    (line,) = cv.ax.plot(xs, ys, color=color, linewidth=cv.px(width) * 72 / DPI, alpha=alpha,
                         solid_capstyle="round", solid_joinstyle="round", zorder=zorder)
    return line


# ---------------------------------------------------------------- export
def save_animation(frames, out_base, fps=FPS, gif=True, gif_width=960):
    """Write <out_base>.mp4 (H.264), <out_base>_final.png and optionally <out_base>.gif."""
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    mp4 = out_base.with_suffix(".mp4")
    proc, last = None, None
    for f in frames:
        if proc is None:
            h, w = f.shape[:2]
            proc = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                 "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                 "-crf", "18", "-preset", "slow", "-movflags", "+faststart", str(mp4)],
                stdin=subprocess.PIPE)
        proc.stdin.write(np.ascontiguousarray(f).tobytes())
        last = f
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed writing {mp4}")
    png = out_base.parent / f"{out_base.name}_final.png"
    Image.fromarray(last).save(png)
    written = [mp4, png]
    if gif:
        g = out_base.with_suffix(".gif")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4), "-vf",
                        f"scale={gif_width}:-1:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];"
                        "[b][p]paletteuse=dither=bayer:bayer_scale=4", str(g)], check=True)
        written.append(g)
    return written
