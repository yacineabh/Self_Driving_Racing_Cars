import math
import torch, torch.nn as nn

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0

def mlp(i, h, o, n_hidden=2, act=nn.ReLU):
    """Build a fully-connected MLP with LayerNorm after each hidden layer."""
    layers, d = [], i
    for _ in range(n_hidden):
        layers += [nn.Linear(d, h), nn.LayerNorm(h), act()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)

def split_gaussian(out):
    mean, log_std = out.chunk(2, dim=-1)
    return mean, log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)


# ─────────────────────────────────────────────────────────────────
#  Encoder
# ─────────────────────────────────────────────────────────────────

class LidarEncoder(nn.Module):
    """state (B, [T,] S) -> embedding (B, [T,] E)

    Input is the augmented state vector (54 LiDAR beams + speed + 2 prev-action
    dims = 57 by default).  A plain MLP with LayerNorm is used so that the
    embedding cannot collapse to a constant (a fixed raw-state prediction target
    prevents that separately for TAWM, but the encoder still benefits from
    normalisation layers).
    """
    def __init__(self, state_dim, embed_dim, hidden):
        super().__init__()
        self.net = mlp(state_dim, hidden, embed_dim)

    def forward(self, s):
        return self.net(s)


# ─────────────────────────────────────────────────────────────────
#  Skill posterior  q(z | tau)
# ─────────────────────────────────────────────────────────────────

class SkillPosterior(nn.Module):
    """q(z | tau): bidirectional GRU over (embedding, action) pairs -> Gaussian z.

    The bi-GRU reads the whole window of T (state-embedding, action) pairs and
    compresses them into a single latent Gaussian.  Using the full trajectory
    (not just the start state) gives the posterior access to *what actually
    happened* so it can infer which skill was executed.
    """
    def __init__(self, embed_dim, action_dim, z_dim, hidden):
        super().__init__()
        self.inp = nn.Linear(embed_dim + action_dim, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * hidden, 2 * z_dim)

    def forward(self, e_seq, a_seq):          # (B, T, E), (B, T, A)
        x = torch.relu(self.inp(torch.cat([e_seq, a_seq], -1)))
        _, h = self.gru(x)                    # h: (2, B, H)  fwd + bwd final hidden
        return split_gaussian(self.out(torch.cat([h[0], h[1]], -1)))


# ─────────────────────────────────────────────────────────────────
#  Skill prior  p(z | s0)
# ─────────────────────────────────────────────────────────────────

class SkillPrior(nn.Module):
    """p(z | s0): skill distribution given only the start state embedding."""
    def __init__(self, embed_dim, z_dim, hidden):
        super().__init__()
        self.net = mlp(embed_dim, hidden, 2 * z_dim)

    def forward(self, e0):
        return split_gaussian(self.net(e0))


# ─────────────────────────────────────────────────────────────────
#  Low-level policy  pi(a | e_t, z)
# ─────────────────────────────────────────────────────────────────

class LowLevelPolicy(nn.Module):
    """pi(a | e_t, z): one skill decodes into a whole action sequence.

    z is broadcast over the time axis so the same latent guides every tick.
    """
    def __init__(self, embed_dim, z_dim, action_dim, hidden):
        super().__init__()
        self.net = mlp(embed_dim + z_dim, hidden, 2 * action_dim)

    def forward(self, e, z):
        if e.dim() == 3:                              # (B, T, E): broadcast z over time
            z = z.unsqueeze(1).expand(-1, e.size(1), -1)
        return split_gaussian(self.net(torch.cat([e, z], -1)))


# ─────────────────────────────────────────────────────────────────
#  Temporally Abstract World Model  (TAWM)
# ─────────────────────────────────────────────────────────────────

class TAWM(nn.Module):
    """(e0, z) -> predicted change in raw state after T steps
                + metres of progress
                + normalised minimum LiDAR range over the horizon  [NEW]

    Predicting the *raw* LiDAR delta (not the embedding delta) prevents the
    encoder from collapsing to a constant.  The progress head supplies the
    reward signal for the planner.  The min-range head gives the planner a
    cheap, jointly-trained safety signal without a separate model.
    """
    def __init__(self, embed_dim, z_dim, state_dim, hidden):
        super().__init__()
        # + 1 for progress, + 1 for min_range = state_dim + 2 outputs
        self.net = mlp(embed_dim + z_dim, hidden, state_dim + 2)
        self.state_dim = state_dim

    def forward(self, e0, z):
        out = self.net(torch.cat([e0, z], -1))       # (B, state_dim + 2)
        delta_state = out[..., :self.state_dim]       # (B, S)
        progress    = out[..., -2]                    # (B,)
        min_range   = torch.sigmoid(out[..., -1])     # (B,)  -> [0, 1]
        return delta_state, progress, min_range