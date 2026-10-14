"""Tabular planning and learning on a GridWorld.

Values are dicts {state: float}, policies {state: {action: prob}}, Q-tables {(state, action): float}.
"""
import numpy as np

from gridworld import ACTIONS


# ---------------------------------------------------------------- planning
def q_from_v(env, V, s, a):
    return sum(p * (r + (0.0 if done else env.gamma * V[s2])) for p, s2, r, done in env.outcomes(s, a))


def value_iteration(env, tol=1e-10, max_sweeps=1000):
    """Returns V after every sweep; history[0] is all zeros."""
    V = {s: 0.0 for s in env.states}
    history = [V]
    for _ in range(max_sweeps):
        V = {s: 0.0 if env.is_terminal(s) else max(q_from_v(env, history[-1], s, a) for a in ACTIONS)
             for s in env.states}
        history.append(V)
        if max(abs(V[s] - history[-2][s]) for s in env.states) < tol:
            break
    return history


def policy_evaluation(env, pi, tol=1e-12, max_sweeps=100000):
    V = {s: 0.0 for s in env.states}
    for _ in range(max_sweeps):
        V_new = {s: 0.0 if env.is_terminal(s) else
                 sum(p_a * q_from_v(env, V, s, a) for a, p_a in pi[s].items())
                 for s in env.states}
        delta = max(abs(V_new[s] - V[s]) for s in env.states)
        V = V_new
        if delta < tol:
            break
    return V


def optimal_actions(env, V, tol=1e-9):
    """All greedy actions per state, ties kept."""
    out = {}
    for s in env.states:
        if env.is_terminal(s):
            continue
        qs = [q_from_v(env, V, s, a) for a in ACTIONS]
        out[s] = [a for a, q in zip(ACTIONS, qs) if q >= max(qs) - tol]
    return out


def greedy_policy(env, V):
    return {s: {acts[0]: 1.0} for s, acts in optimal_actions(env, V).items()}


def random_policy(env):
    return {s: {a: 1 / len(ACTIONS) for a in ACTIONS} for s in env.states if not env.is_terminal(s)}


# ---------------------------------------------------------------- simulation
def rollout(env, pi, rng, slip=None, start=None, max_steps=200):
    """One episode as a list of (s, a, r, s2, done, slipped)."""
    s = env.start if start is None else start
    steps = []
    for _ in range(max_steps):
        acts, probs = zip(*pi[s].items())
        a = acts[rng.choice(len(acts), p=probs)] if len(acts) > 1 else acts[0]
        s2, r, done, slipped = env.step(s, a, rng, slip=slip)
        steps.append((s, a, r, s2, done, slipped))
        if done:
            break
        s = s2
    return steps


def discounted_return(rewards, gamma):
    return sum(gamma ** k * r for k, r in enumerate(rewards))


def returns_to_go(rewards, gamma):
    """[G_0, G_1, ...] with G_t = r_{t+1} + gamma G_{t+1}."""
    G, out = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G
        out.append(G)
    return out[::-1]


# ---------------------------------------------------------------- learning
def epsilon_greedy(Q, s, eps, rng):
    if rng.random() < eps:
        return ACTIONS[rng.integers(len(ACTIONS))]
    qs = np.array([Q[(s, a)] for a in ACTIONS])
    best = np.flatnonzero(qs >= qs.max() - 1e-12)
    return ACTIONS[best[rng.integers(len(best))]]


def _learn(env, episodes, epsilon, seed, max_steps, update_step, update_episode,
           snapshot_at, record_at, record_steps_of):
    rng = np.random.default_rng(seed)
    Q = {(s, a): 0.0 for s in env.states if not env.is_terminal(s) for a in ACTIONS}
    N = {k: 0 for k in Q}
    visited = {env.start}
    out = dict(lengths=[], snapshots={}, episodes={}, step_snapshots={})
    for ep in range(1, episodes + 1):
        eps = epsilon(ep) if callable(epsilon) else epsilon
        s, steps, per_step = env.start, [], []
        for _ in range(max_steps):
            a = epsilon_greedy(Q, s, eps, rng)
            s2, r, done, slipped = env.step(s, a, rng)
            steps.append((s, a, r, s2, done, slipped))
            if update_step:
                update_step(Q, N, s, a, r, s2, done)
                visited.add(s2)
            if ep in record_steps_of:
                per_step.append((dict(Q), set(visited)))
            if done:
                break
            s = s2
        if update_episode:
            update_episode(Q, N, steps)
            visited.update(st[0] for st in steps)
        out["lengths"].append(len(steps))
        if ep in snapshot_at:
            out["snapshots"][ep] = (dict(Q), set(visited))
        if ep in record_at:
            out["episodes"][ep] = steps
        if ep in record_steps_of:
            out["step_snapshots"][ep] = per_step
    out.update(Q=Q, N=N, visited=visited)
    return out


def q_learning(env, episodes, epsilon=0.2, omega=0.7, seed=0, max_steps=100,
               snapshot_at=(), record_at=(), record_steps_of=()):
    """Tabular Q-learning, epsilon-greedy, step size alpha = 1 / N(s, a)^omega.

    snapshot_at: episodes after which (Q, visited) is stored.
    record_at: episodes whose steps are stored.
    record_steps_of: episodes for which (Q, visited) is stored after every step.
    """
    def update(Q, N, s, a, r, s2, done):
        N[(s, a)] += 1
        target = r if done else r + env.gamma * max(Q[(s2, b)] for b in ACTIONS)
        Q[(s, a)] += (target - Q[(s, a)]) / N[(s, a)] ** omega

    return _learn(env, episodes, epsilon, seed, max_steps, update, None,
                  set(snapshot_at), set(record_at), set(record_steps_of))


def mc_control(env, episodes, epsilon=lambda k: 1 / k, seed=0, max_steps=100,
               snapshot_at=(), record_at=(), record_steps_of=()):
    """Every-visit Monte Carlo control: after each episode, Q(s_t, a_t) moves to the running mean of G_t."""
    def update(Q, N, steps):
        for (s, a, *_), G in zip(steps, returns_to_go([st[2] for st in steps], env.gamma)):
            N[(s, a)] += 1
            Q[(s, a)] += (G - Q[(s, a)]) / N[(s, a)]

    return _learn(env, episodes, epsilon, seed, max_steps, None, update,
                  set(snapshot_at), set(record_at), set(record_steps_of))


def max_q(env, Q):
    return {s: max(Q[(s, a)] for a in ACTIONS) for s in env.states if not env.is_terminal(s)}


def greedy_actions(env, Q, visited=None, tol=1e-9):
    """Best actions per state from Q; states with no preference yet (or unvisited) are left out."""
    out = {}
    for s in env.states:
        if env.is_terminal(s) or (visited is not None and s not in visited):
            continue
        qs = [Q[(s, a)] for a in ACTIONS]
        best = [a for a, q in zip(ACTIONS, qs) if q >= max(qs) - tol]
        if len(best) < len(ACTIONS):
            out[s] = best
    return out


def path(env, pi, s=None, max_len=100):
    """States visited by a deterministic policy, assuming no slip."""
    s = env.start if s is None else s
    out = [s]
    while not env.is_terminal(s) and len(out) <= max_len:
        s = env.move(s, next(iter(pi[s])))
        out.append(s)
    return out
