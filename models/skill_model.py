import math
import torch, torch.nn as nn
from models.networks import LidarEncoder, SkillPosterior, SkillPrior, LowLevelPolicy, TAWM

# ── Constants ─────────────────────────────────────────────────────────────────
PROGRESS_SCALE = 3.0   # metres; targets divided by this → ~O(1)
N_LIDAR_BEAMS  = 54    # first 54 dims of state = normalised LiDAR beams
RAW_STATE_DIM  = 55    # LiDAR beams + speed  (NO prev-action dims)
                       # dims 55-56 of the 57-dim state are prev_action (info
                       # for the encoder/posterior, NOT a TAWM prediction target)

# ── helpers ───────────────────────────────────────────────────────────────────

def gaussian_nll(x, mu, log_std):
    """Element-wise Gaussian NLL."""
    return 0.5 * ((x - mu) / log_std.exp()) ** 2 + log_std + 0.5 * math.log(2 * math.pi)

def kl_diag(mu_q, ls_q, mu_p, ls_p):
    """Analytic KL[ q(z) || p(z) ] for diagonal Gaussians."""
    return (ls_p - ls_q) + (torch.exp(2 * ls_q) + (mu_q - mu_p) ** 2) / (2 * torch.exp(2 * ls_p)) - 0.5


# ── SkillModel ────────────────────────────────────────────────────────────────

class SkillModel(nn.Module):
    """Offline, skill-conditioned world model (OPOSM / SelfRace variant).

    Encoder input  : 57-dim augmented state  (54 LiDAR beams + speed + 2 prev-action)
    TAWM target    : 55-dim RAW state delta  (LiDAR beams + speed ONLY)
                     Prev-action dims are deliberately EXCLUDED from the TAWM
                     target.  Predicting action-change is already the policy's
                     job and including it inflates the TAWM baseline error by
                     making the loss heterogeneous in scale.

    Components
    ----------
    encoder   : state (57) -> embedding (64)
    posterior : q(z | tau)  bi-GRU over (embedding, action) -> Gaussian z (16)
    prior     : p(z | s0)   MLP from start embedding -> Gaussian z
    policy    : pi(a | e, z) embedding + z -> Gaussian action (2)
    tawm      : (e0, z) -> (delta_raw_state (55), progress, min_range)

    Loss
    ----
    total = policy_nll
          + tawm_weight * (tawm_state_mse + tawm_prog_mse + range_weight * tawm_range_mse)
          + kl_beta * KL(q || p)
    """

    def __init__(self, state_dim=57, action_dim=2,
                 embed_dim=64, z_dim=16, hidden=256,
                 kl_beta=1.0, tawm_weight=1.0, range_weight=0.5):
        super().__init__()
        self.encoder   = LidarEncoder(state_dim, embed_dim, hidden)
        self.posterior = SkillPosterior(embed_dim, action_dim, z_dim, hidden)
        self.prior     = SkillPrior(embed_dim, z_dim, hidden)
        self.policy    = LowLevelPolicy(embed_dim, z_dim, action_dim, hidden)
        # TAWM predicts the RAW state delta (55-dim), NOT the full 57-dim delta.
        self.tawm      = TAWM(embed_dim, z_dim, RAW_STATE_DIM, hidden)
        self.kl_beta   = kl_beta
        self.tawm_weight  = tawm_weight
        self.range_weight = range_weight

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, states, actions, progress, freeze_posterior=False):
        """
        Parameters
        ----------
        states   : (B, T+1, 57)   augmented state
        actions  : (B, T,   2)    normalised actions
        progress : (B, T+1)       cumulative metres along centreline

        Returns dict of scalar losses.
        """
        e    = self.encoder(states)   # (B, T+1, 64)
        e_in = e[:, :-1]              # (B, T,   64)  paired with actions

        # ── posterior / prior ─────────────────────────────────────────────────
        if freeze_posterior:
            with torch.no_grad():
                mu_q, ls_q = self.posterior(e_in.detach(), actions)
        else:
            mu_q, ls_q = self.posterior(e_in, actions)

        mu_p, ls_p = self.prior(e[:, 0])
        z = mu_q + ls_q.exp() * torch.randn_like(mu_q)   # reparametrised

        # ── policy NLL ────────────────────────────────────────────────────────
        a_mu, a_ls = self.policy(e_in, z)
        policy_nll  = gaussian_nll(actions, a_mu, a_ls).sum((1, 2)).mean()

        # ── TAWM on RAW state (first 55 dims only) ────────────────────────────
        ds_pred, prog_pred, range_pred = self.tawm(e[:, 0], z)

        # Target: change in the ORIGINAL sensor observation (beams + speed).
        # We deliberately exclude dims 55-56 (prev_action) from the TAWM target.
        ds_true    = states[:, -1, :RAW_STATE_DIM] - states[:, 0, :RAW_STATE_DIM]
        prog_true  = (progress[:, -1] - progress[:, 0]) / PROGRESS_SCALE
        # Min normalised beam range observed across the whole window
        range_true = states[:, :, :N_LIDAR_BEAMS].min(dim=1).values.min(dim=1).values

        tawm_state  = ((ds_pred  - ds_true)  ** 2).sum(-1).mean()
        tawm_prog   = ((prog_pred - prog_true) ** 2).mean()
        tawm_range  = ((range_pred - range_true) ** 2).mean()
        tawm_loss   = tawm_state + tawm_prog + self.range_weight * tawm_range

        # ── KL ────────────────────────────────────────────────────────────────
        kl = kl_diag(mu_q, ls_q, mu_p, ls_p).sum(-1).mean()

        total = policy_nll + self.tawm_weight * tawm_loss + self.kl_beta * kl

        with torch.no_grad():
            # "predict no change" baseline for the 55-dim raw state
            baseline = (ds_true ** 2).sum(-1).mean()

        return {"total": total, "policy_nll": policy_nll, "kl": kl,
                "tawm_state": tawm_state, "tawm_prog": tawm_prog,
                "tawm_range": tawm_range, "tawm_baseline": baseline}

    # ── diagnostics ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def diagnostics(self, states, actions, progress):
        """Z-usage and TAWM quality metrics on a batch (no gradient)."""
        e     = self.encoder(states)
        e_in  = e[:, :-1]
        mu_q, _    = self.posterior(e_in, actions)
        mu_p, ls_p = self.prior(e[:, 0])
        ds_true    = states[:, -1, :RAW_STATE_DIM] - states[:, 0, :RAW_STATE_DIM]
        range_true = states[:, :, :N_LIDAR_BEAMS].min(dim=1).values.min(dim=1).values

        def tawm_err(z):
            ds, _, _ = self.tawm(e[:, 0], z)
            return ((ds - ds_true) ** 2).sum(-1).mean().item()

        def pol_nll(z):
            m, s = self.policy(e_in, z)
            return gaussian_nll(actions, m, s).sum((1, 2)).mean().item()

        def range_err(z):
            _, _, r = self.tawm(e[:, 0], z)
            return ((r - range_true) ** 2).mean().item()

        z_post  = mu_q                                         # "right" z for this window
        z_prior = mu_p + ls_p.exp() * torch.randn_like(mu_p)  # what planner will sample
        z_shuf  = z_post[torch.randperm(z_post.size(0))]      # z from unrelated window

        return {
            "tawm_baseline":   (ds_true ** 2).sum(-1).mean().item(),
            "tawm_post":       tawm_err(z_post),
            "tawm_prior":      tawm_err(z_prior),
            "tawm_shuf":       tawm_err(z_shuf),
            "pol_post":        pol_nll(z_post),
            "pol_prior":       pol_nll(z_prior),
            "pol_shuf":        pol_nll(z_shuf),
            "range_err_post":  range_err(z_post),
        }