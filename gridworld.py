"""Grid world with a slippery puddle.

States are (row, col) cells, (0, 0) is top-left. Entering G gives +10 and ends the episode.
Any action taken while standing on the puddle slips with probability SLIP_PROB:
reward -10 and the episode ends.
"""
import numpy as np

LAYOUTS = {
    "small": """\
# . . ~ G
. . # . .
S . # . .
. . . . .""",
    "large": """\
# . . . ~ G
. . # # . .
S . # # . .
. . . . . .
# . . . . .""",
}

ACTIONS = ("up", "right", "down", "left")
DELTAS = {"up": (-1, 0), "right": (0, 1), "down": (1, 0), "left": (0, -1)}

GAMMA = 0.9
SLIP_PROB = 0.2
GOAL_REWARD = 10.0
SLIP_REWARD = -10.0


class GridWorld:
    def __init__(self, layout="small", gamma=GAMMA, slip_prob=SLIP_PROB):
        self.layout = layout
        self.grid = [row.split() for row in LAYOUTS[layout].splitlines()]
        self.rows, self.cols = len(self.grid), len(self.grid[0])
        self.gamma, self.slip_prob = gamma, slip_prob
        self.start = self._find("S")
        self.puddle = self._find("~")
        self.goal = self._find("G")
        self.walls = frozenset(self._cells("#"))
        self.states = [(r, c) for r in range(self.rows) for c in range(self.cols)
                       if (r, c) not in self.walls]

    def _cells(self, ch):
        return [(r, c) for r in range(self.rows) for c in range(self.cols) if self.grid[r][c] == ch]

    def _find(self, ch):
        return self._cells(ch)[0]

    def is_terminal(self, s):
        return s == self.goal

    def move(self, s, a):
        r, c = s[0] + DELTAS[a][0], s[1] + DELTAS[a][1]
        if not (0 <= r < self.rows and 0 <= c < self.cols) or (r, c) in self.walls:
            return s
        return (r, c)

    def outcomes(self, s, a):
        """List of (prob, next_state, reward, done)."""
        if self.is_terminal(s):
            return [(1.0, s, 0.0, True)]
        s2 = self.move(s, a)
        normal = (s2, GOAL_REWARD, True) if s2 == self.goal else (s2, 0.0, False)
        if s != self.puddle or self.slip_prob == 0:
            return [(1.0, *normal)]
        return [(1.0 - self.slip_prob, *normal), (self.slip_prob, self.puddle, SLIP_REWARD, True)]

    def expected_reward(self, s, a):
        return sum(p * r for p, _, r, _ in self.outcomes(s, a))

    def step(self, s, a, rng, slip=None):
        """Sample one step -> (next_state, reward, done, slipped). `slip` forces the puddle outcome."""
        outs = self.outcomes(s, a)
        if len(outs) == 1:
            s2, r, done = outs[0][1:]
            return s2, r, done, False
        slipped = (rng.random() < self.slip_prob) if slip is None else bool(slip)
        _, s2, r, done = outs[1] if slipped else outs[0]
        return s2, r, done, slipped


def make_rng(seed=0):
    return np.random.default_rng(seed)
