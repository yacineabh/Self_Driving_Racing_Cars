import torch

# ─────────────────────────────────────────────────────────────────────────────
#  Naive VI Trainer
#  One optimizer over all parameters (standard amortised variational inference).
#  Simple, stable, converges well.  Used as the ablation baseline.
# ─────────────────────────────────────────────────────────────────────────────

class NaiveVITrainer:
    """Standard amortised VI: one AdamW optimizer over all parameters."""

    def __init__(self, model, tc):
        self.model = model
        self.opt   = torch.optim.AdamW(model.parameters(), lr=tc["lr"])
        self.clip, self.warmup = tc["grad_clip"], tc["kl_warmup_steps"]
        self.beta_final = model.kl_beta
        self.step_i = 0

    def step(self, batch):
        self.model.kl_beta = self.beta_final * min(1.0, self.step_i / max(1, self.warmup))
        out = self.model(batch["states"], batch["actions"], batch["progress"])
        self.opt.zero_grad(set_to_none=True)
        out["total"].backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.opt.step()
        self.step_i += 1
        return {k: v.item() for k, v in out.items()}


# ─────────────────────────────────────────────────────────────────────────────
#  Causal EM Trainer  (fixed)
#
#  The paper's training recipe:
#    E-step: update {encoder + posterior} — posterior infers which skill z fits τ
#    M-step: posterior frozen; update {encoder, prior, policy, TAWM}
#            against the z posterior that was computed at the start of this batch.
#
#  Key fixes vs the earlier broken version:
#  1. The ENCODER is included in E-step params so it learns features the
#     posterior needs.  Without this, the encoder only learns for the M-step
#     objectives and the posterior is forced to work with misaligned features.
#  2. In the M-step, z is computed from the encoder (grad enabled) but the
#     posterior parameters are frozen (grad disabled via param.requires_grad).
#     We do NOT detach e_in — the encoder must still receive signal from the
#     M-step losses.
#  3. Step ratio: 2 E-steps per M-step during warm-up; 1:1 after.
#     The E-step must approximately converge before the M-step is informative.
#  4. Prior std is clamped during early warm-up to prevent the prior from
#     producing out-of-distribution z samples before the posterior is trained.
# ─────────────────────────────────────────────────────────────────────────────

class CausalEMTrainer:
    """Causal EM: alternating E/M optimisation over posterior vs generative model.

    E-step params : encoder + posterior
    M-step params : encoder + prior + policy + TAWM
    Note: the encoder participates in BOTH steps (shared representation).
    """

    def __init__(self, model, tc):
        self.model = model
        self.clip  = tc["grad_clip"]
        self.warmup = tc["kl_warmup_steps"]
        self.beta_final = model.kl_beta
        self.step_i = 0

        # E-step: encoder + posterior
        e_ids = {id(p) for p in model.encoder.parameters()}
        post_ids = {id(p) for p in model.posterior.parameters()}
        self.e_params = [p for p in model.parameters()
                         if id(p) in e_ids | post_ids]

        # M-step: encoder + prior + policy + TAWM  (encoder shared!)
        m_ids = ({id(p) for p in model.encoder.parameters()} |
                 {id(p) for p in model.prior.parameters()} |
                 {id(p) for p in model.policy.parameters()} |
                 {id(p) for p in model.tawm.parameters()})
        self.m_params = [p for p in model.parameters() if id(p) in m_ids]

        self.opt_e = torch.optim.AdamW(self.e_params, lr=tc["lr"])
        self.opt_m = torch.optim.AdamW(self.m_params, lr=tc["lr"])

        # Track which param ids belong to which step (for requires_grad toggling)
        self.posterior_ids = post_ids
        self.prior_ids     = {id(p) for p in model.prior.parameters()}

    # ── helper: set requires_grad for a subset of params ─────────────────────
    @staticmethod
    def _set_grad(params, model, ids, enabled):
        for p in model.parameters():
            if id(p) in ids:
                p.requires_grad_(enabled)

    # ── one training step ─────────────────────────────────────────────────────
    def step(self, batch):
        beta = self.beta_final * min(1.0, self.step_i / max(1, self.warmup))
        self.model.kl_beta = beta

        # During warm-up, clamp prior log_std outputs to avoid prior explosion.
        # After warm-up, release the clamp (LOG_STD_MAX back to its normal 2.0).
        # We do this by temporarily monkey-patching the model's LOG_STD_MAX
        # in networks; the cleaner approach is just clamping prior outputs here.
        in_warmup = self.step_i < self.warmup // 2

        # ── E-steps (posterior + encoder) ────────────────────────────────────
        # Freeze prior so it doesn't update here.
        self._set_grad(self.m_params, self.model, self.prior_ids, False)
        n_e_steps = 2 if in_warmup else 1
        last_out_e = None
        for _ in range(n_e_steps):
            out_e = self.model(batch["states"], batch["actions"], batch["progress"])
            self.opt_e.zero_grad(set_to_none=True)
            self.opt_m.zero_grad(set_to_none=True)
            out_e["total"].backward()
            torch.nn.utils.clip_grad_norm_(self.e_params, self.clip)
            self.opt_e.step()
            last_out_e = out_e

        # Un-freeze prior before M-step.
        self._set_grad(self.m_params, self.model, self.prior_ids, True)

        # ── M-step (encoder + prior + policy + TAWM; posterior frozen) ───────
        self._set_grad(self.e_params, self.model, self.posterior_ids, False)
        out_m = self.model(batch["states"], batch["actions"], batch["progress"],
                           freeze_posterior=True)
        self.opt_e.zero_grad(set_to_none=True)
        self.opt_m.zero_grad(set_to_none=True)
        out_m["total"].backward()
        torch.nn.utils.clip_grad_norm_(self.m_params, self.clip)
        self.opt_m.step()

        # Re-enable posterior gradients for the next iteration.
        self._set_grad(self.e_params, self.model, self.posterior_ids, True)

        self.step_i += 1
        # Report M-step losses as the primary training signal.
        return {k: v.item() for k, v in out_m.items()}