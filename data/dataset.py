import pickle, numpy as np, torch
from torch.utils.data import Dataset


class OfflineSkillDataset(Dataset):
    """Sliding windows of T actions and T+1 states, never crossing episode
    boundaries.  Windows that contain any incomplete (collision mid-tick)
    action are silently skipped via the ``valid_actions`` mask collected by
    the updated collector."""

    def __init__(self, paths=None, horizon=20, stride=2,
                 action_norm=None, episodes=None):
        self.T, self.stride, self.action_norm = horizon, stride, action_norm

        if episodes is None:
            episodes = []
            for p in paths:
                batch = pickle.load(open(p, "rb"))
                episodes += batch["episodes"]
        self.episodes = episodes

        # Build index: (episode_idx, window_start)
        # Only include windows where every action in [start, start+T) is valid.
        self.index = []
        for e, ep in enumerate(self.episodes):
            n_actions = len(ep["actions"])
            valid = np.asarray(
                ep.get("valid_actions", np.ones(n_actions, bool)), bool)
            assert len(valid) == n_actions, \
                f"episode {e}: valid_actions length mismatch"
            for s in range(0, n_actions - horizon + 1, stride):
                if valid[s:s + horizon].all():
                    self.index.append((e, s))

    # ------------------------------------------------------------------ #
    def split(self, val_frac, seed=0):
        """Episode-level train/val split (never splits within an episode)."""
        perm = np.random.default_rng(seed).permutation(len(self.episodes))
        n_val = max(1, int(len(perm) * val_frac))
        kw = dict(horizon=self.T, stride=self.stride,
                  action_norm=self.action_norm)
        val_eps = [self.episodes[i] for i in perm[:n_val]]
        tr_eps  = [self.episodes[i] for i in perm[n_val:]]
        return (OfflineSkillDataset(episodes=tr_eps,  **kw),
                OfflineSkillDataset(episodes=val_eps, **kw))

    # ------------------------------------------------------------------ #
    def all_actions(self):
        """Return all *valid* raw actions (used to fit the action normalizer)."""
        parts = []
        for ep in self.episodes:
            n = len(ep["actions"])
            valid = np.asarray(ep.get("valid_actions", np.ones(n, bool)), bool)
            if valid.any():
                parts.append(ep["actions"][valid])
        return np.concatenate(parts) if parts else np.zeros((0, 2), np.float32)

    # ------------------------------------------------------------------ #
    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        e, s = self.index[i]
        ep, T = self.episodes[e], self.T

        # Actions for the window
        a_raw = ep["actions"][s:s + T]
        if self.action_norm is not None:
            a = self.action_norm(a_raw).astype(np.float32)
        else:
            a = a_raw.astype(np.float32)

        # Previous actions: action at tick t-1 (zeros for the first tick of the
        # window that falls on the episode boundary, i.e. s == 0).
        if "prev_actions" in ep:
            prev_a_raw = ep["prev_actions"][s:s + T]
        else:
            # Backwards-compat with old datasets that lack prev_actions:
            # reconstruct from the actions array.
            prev_a_raw = np.zeros_like(a_raw)
            if s > 0:
                prev_a_raw[0] = ep["actions"][s - 1]
            prev_a_raw[1:] = a_raw[:-1]

        if self.action_norm is not None:
            prev_a = self.action_norm(prev_a_raw).astype(np.float32)
        else:
            prev_a = prev_a_raw.astype(np.float32)

        # Concatenate previous action onto each state frame: 55 + 2 = 57 dims.
        # states shape (T+1, 55); we need prev_a at each of the T+1 positions.
        # For position t in [0, T], the previous action is:
        #   t == 0: zeros (or action[s-1] when s > 0, handled above for t=0)
        #   t >= 1: action[s + t - 1]
        # We already built prev_a for t in [0, T-1].  For t=T we use action[s+T-1].
        states_raw = ep["states"][s:s + T + 1]                          # (T+1, 55)
        last_prev = a_raw[-1:].copy()                                    # action before tick T
        if self.action_norm is not None:
            last_prev = self.action_norm(last_prev).astype(np.float32)
        prev_a_full = np.concatenate([prev_a, last_prev], axis=0)       # (T+1, 2)
        states_aug = np.concatenate(                                     # (T+1, 57)
            [states_raw, prev_a_full], axis=-1).astype(np.float32)

        return {
            "states":   torch.from_numpy(states_aug),                   # (T+1, 57)
            "actions":  torch.from_numpy(a),                            # (T,   2)
            "progress": torch.from_numpy(
                ep["progress"][s:s + T + 1]),                           # (T+1,)
        }