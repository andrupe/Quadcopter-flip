"""
Causal recurrent history encoder over the observation stream.

WHY A GRU AND NOT THE TCN THIS REPLACED
A dilated causal TCN is a better offline model per parameter, and the version used here
previously was correct and fully causal. It is the wrong shape for this vehicle. A TCN
needs the last `receptive_field` frames available at every inference, so the per-step cost
is O(RF) in both compute and RAM (253 frames x 21 features x 4 bytes = 21 KB of history
buffer, plus 63 dilated convolutions per step). The STM32F405 on a Crazyflie 2.1 has
192 KB of RAM, no NEON, and roughly 100 MMAC/s, so a 253-tap stack is not a comfortable
onboard budget. A GRU carries a single hidden vector instead: the per-step cost is
O(hidden^2), the state is one 48-float tensor, and the cost does not grow with how far
back the model looks.

  hidden = 48, f_in = 33, z = 16  ->  ~15.1k parameters
  per-step cost ~ 3*(48*33 + 48*48) = ~11.7k MACs -> ~0.12 ms at 100 MMAC/s

That is a comfortable fit for 50-100 Hz, and it is O(1) in episode length by construction
rather than by truncating the window.

STILL STRICTLY CAUSAL. A GRU consumes inputs left to right and the hidden state at step t
is a function of inputs <= t only. There is no bidirectionality anywhere, so supervising
every timestep of a whole episode in one pass is safe - and that is exactly how the
training script should supervise it.

TRAINING MUST BE EPISODE-SEQUENTIAL. The TCN could be trained on randomly sampled windows
with the history prefilled, because its receptive field bounded what mattered. A GRU's
state is an unbounded summary of everything it has seen, so a randomly-posed window gives
it a state it would never actually be in. `forward_sequence` therefore takes whole
episodes and returns the hidden state at every step; the loss is taken over the whole
episode, and the state is carried in from the true start.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .observation_spec import ENCODER_IN_DIM

VAR_MIN: float = -6.0
VAR_MAX: float = 4.0


class HistoryEncoder(nn.Module):
    """
    [B, T, f_in] -> [B, z_dim] (last step) or [B, T, z_dim] (all steps).

    Single-layer GRU on purpose: the hidden state is then ONE tensor, which keeps the
    incremental API trivial and means the deployed state is a single 48-float buffer.
    Stacking layers would buy capacity the per-step budget does not need and would make
    the runtime state a tuple.

    `dilations`/`kernel`/`groups` are accepted and IGNORED so that checkpoints and call
    sites written against the old TCN keep loading; they were architectural choices of a
    model that no longer exists.
    """

    def __init__(
        self,
        f_in: int = ENCODER_IN_DIM,
        width: int = 48,
        z_dim: int = 16,
        dilations: Optional[Sequence[int]] = None,
        kernel: int = 3,
        groups: int = 1,
    ):
        super().__init__()
        self.f_in = int(f_in)
        self.width = int(width)          # GRU hidden size; kept named `width` for compat
        self.z_dim = int(z_dim)
        # Retained only so an old checkpoint config round-trips.
        self.dilations = tuple(int(d) for d in (dilations or ()))

        self.gru = nn.GRU(self.f_in, self.width, num_layers=1, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(self.width, self.width),
            nn.GELU(),
            nn.Linear(self.width, self.z_dim),
            nn.Tanh(),
        )

    @property
    def receptive_field(self) -> int:
        """
        Not an integer for a GRU, and that is the point.

        A TCN has a hard lookback horizon; a GRU's state is an unbounded (and learned)
        summary of the whole episode. Reporting 0 rather than a fake number keeps any
        caller that used this to size a history buffer from silently sizing it wrongly.
        """
        return 0

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # -- batch API -----------------------------------------------------------------
    def _check(self, x: torch.Tensor) -> None:
        if x.dim() != 3:
            raise ValueError(f"expected [B, T, f_in], got {tuple(x.shape)}")
        if x.shape[-1] != self.f_in:
            raise ValueError(f"expected f_in={self.f_in}, got {x.shape[-1]}")

    def forward_sequence(self, x: torch.Tensor, h0: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Readout at every timestep: [B, T, f_in] -> [B, T, z_dim].

        `h0` lets a caller continue an episode across chunks, which is how a training
        script keeps the state consistent when it splits a long episode into batches.
        """
        self._check(x)
        out, _ = self.gru(x, self._init_h(x.shape[0], x.device) if h0 is None else h0.unsqueeze(0))
        return self.head(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Readout at the final timestep: [B, T, f_in] -> [B, z_dim]."""
        return self.forward_sequence(x)[:, -1]

    # -- incremental API (deployment path) -----------------------------------------
    def _init_h(self, batch: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(1, batch, self.width, device=device)

    def init_state(self, batch: int = 1, device: Optional[torch.device] = None) -> torch.Tensor:
        """Zero hidden state for a fresh episode. Shape [B, hidden]."""
        return self._init_h(batch, device or torch.device("cpu"))[0]

    @torch.no_grad()
    def step(self, x_t: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One control step: ([B, f_in], [B, hidden]) -> ([B, z_dim], [B, hidden]).

        This is the whole deployed inference cost, and it is independent of how long the
        episode has been running - which the TCN's 253-frame window was not.
        """
        if x_t.dim() != 2:
            raise ValueError(f"expected [B, f_in], got {tuple(x_t.shape)}")
        out, h_next = self.gru(x_t.unsqueeze(1), h.unsqueeze(0))
        return self.head(out[:, -1]), h_next[0]


class EncoderWithHead(nn.Module):
    """
    HistoryEncoder plus a heteroscedastic regression head predicting (mu, logvar) for
    every privileged target.

    The per-target variance is not decoration: at minimum it tells you which quantities
    are unidentified, and it can be concatenated to the actor alongside z if the policy
    benefits from knowing how much to trust the estimate.
    """

    def __init__(
        self,
        n_targets: int,
        f_in: int = ENCODER_IN_DIM,
        width: int = 48,
        z_dim: int = 16,
        dilations: Optional[Sequence[int]] = None,
        kernel: int = 3,
        groups: int = 1,
        var_min: float = VAR_MIN,
        var_max: float = VAR_MAX,
    ):
        super().__init__()
        self.n_targets = int(n_targets)
        self.var_min = float(var_min)
        self.var_max = float(var_max)
        self.encoder = HistoryEncoder(
            f_in=f_in, width=width, z_dim=z_dim, dilations=dilations, kernel=kernel, groups=groups
        )
        self.mu_head = nn.Linear(z_dim, self.n_targets)
        self.logvar_head = nn.Linear(z_dim, self.n_targets)

    @property
    def z_dim(self) -> int:
        return self.encoder.z_dim

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward_sequence(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder.forward_sequence(x)         # [B, T, Z]
        return z, self.mu_head(z), self.logvar_head(z)

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        return z, self.mu_head(z), self.logvar_head(z)

    def clamp_logvar(self, logvar: torch.Tensor) -> torch.Tensor:
        return logvar.clamp(self.var_min, self.var_max)


# ---------------------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------------------
def gaussian_nll(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    y: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Heteroscedastic Gaussian NLL: 0.5 * [ (mu - y)^2 / sigma^2 + log sigma^2 ].

    `logvar` is clamped by the caller (EncoderWithHead.clamp_logvar) so that a target the
    encoder cannot identify cannot drive sigma to infinity and collapse the loss.

    `mask` is [B, T] (1 on real timesteps, 0 on sequence padding). Episode batches are
    padded to the longest episode in the batch and the targets of the padded frames are
    zeros, so an unmasked mean fits the variance head to a target value on frames that do
    not exist. The masked form averages over real (timestep, target) elements only.

    CAVEAT, learned the hard way on this project: the loss and R^2 DECOUPLE under this
    objective, because the loss can fall by inflating sigma rather than by improving mu.
    Always log the prediction's standard deviation per group alongside R^2, or a stalled
    encoder looks like a converging one.
    """
    inv_var = torch.exp(-logvar)
    nll = 0.5 * (inv_var * (mu - y) ** 2 + logvar)
    if mask is None:
        return nll.mean()
    m = mask.unsqueeze(-1).to(nll.dtype)
    return (nll * m).sum() / m.sum().clamp(min=1.0) / nll.shape[-1]


def beta_gaussian_nll(mu: torch.Tensor, logvar: torch.Tensor, y: torch.Tensor, beta: float = 0.5) -> torch.Tensor:
    """
    beta-NLL (Seitzer et al., ICLR 2022): 0.5 * (mu - y)^2 / sg(sigma^2)^beta.

    Fallback for when plain NLL misbehaves. beta=0 reduces to MSE on mu; beta=1 is NLL
    without the log term. Caveat: the variance is stop-gradient here, so the logvar head
    receives NO training signal - use this only if you do not need a calibrated
    uncertainty estimate.
    """
    var = torch.exp(logvar)
    weight = var.detach().pow(beta)
    return (0.5 * (mu - y) ** 2 / weight).mean()


def mse_loss(mu: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(mu, y)


# ---------------------------------------------------------------------------------
# checkpoint I/O
#
# Weights and the frozen normalization constants are stored in ONE artifact. A
# mismatched (weights, constants) pair produces no error and no crash - just a silently
# rescaled encoder input - so they are deliberately not free to drift apart.
# ---------------------------------------------------------------------------------
def save_encoder_checkpoint(
    path: str,
    model: "EncoderWithHead",
    norm_stats,
    extra: dict | None = None,
) -> None:
    import os as _os

    payload = {
        "encoder_state": model.encoder.state_dict(),
        "mu_head_state": model.mu_head.state_dict(),
        "logvar_head_state": model.logvar_head.state_dict(),
        "config": {
            "arch": "gru",
            "f_in": model.encoder.f_in,
            "width": model.encoder.width,
            "z_dim": model.encoder.z_dim,
            "n_targets": model.n_targets,
            "var_min": model.var_min,
            "var_max": model.var_max,
            "param_count": model.param_count,
        },
        "norm": norm_stats.to_dict() if hasattr(norm_stats, "to_dict") else dict(norm_stats),
        "extra": extra or {},
    }
    _dir = _os.path.dirname(_os.path.abspath(path))
    _os.makedirs(_dir, exist_ok=True)
    torch.save(payload, path)


def load_encoder_checkpoint(path: str, device: str = "cpu"):
    """
    Returns (encoder, norm_stats, checkpoint_dict).

    `encoder` is a plain HistoryEncoder in eval() mode with gradients disabled - the
    frozen configuration used by LatentObsWrapper.
    """
    from .observation_spec import NormStats

    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    encoder = HistoryEncoder(
        f_in=cfg["f_in"],
        width=cfg["width"],
        z_dim=cfg["z_dim"],
    )
    encoder.load_state_dict(ckpt["encoder_state"])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    norm = NormStats.from_dict(ckpt["norm"])
    return encoder, norm, ckpt
