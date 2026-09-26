"""Cross-version compatibility helpers.

This project was written for the 2023 stack (Python 3.8 / PyTorch 2.1 /
NumPy 1.22).  This module centralises the handful of APIs that have moved or
changed so that the rest of the code base can stay readable.

Target stack: Python 3.12+, PyTorch 2.4 - 2.13, NumPy 2.x.
"""

import os
import warnings

import torch

__all__ = [
    "weights_only_load",
    "torch_load",
    "autocast",
    "make_grad_scaler",
    "weight_norm",
    "remove_weight_norm",
    "torch_version_tuple",
    "silence_known_warnings",
]


def torch_version_tuple():
    """Return ``(major, minor)`` of the running PyTorch."""
    raw = torch.__version__.split("+")[0]
    parts = raw.split(".")
    try:
        return int(parts[0]), int(parts[1])
    except (IndexError, ValueError):  # pragma: no cover - defensive
        return (2, 0)


# --------------------------------------------------------------------------
# Checkpoint loading
# --------------------------------------------------------------------------
def torch_load(path, map_location="cpu", weights_only=None):
    """``torch.load`` that works on both sides of the 2.6 default flip.

    PyTorch >= 2.6 defaults ``weights_only=True``.  The checkpoints produced by
    this project only contain plain python containers plus tensors, so the safe
    path works; but older / third-party checkpoints may carry extra objects.
    We therefore try the safe path first and fall back with a clear warning.
    """
    if weights_only is not None:
        return torch.load(path, map_location=map_location, weights_only=weights_only)

    major, minor = torch_version_tuple()
    if (major, minor) < (2, 6):
        return torch.load(path, map_location=map_location)

    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception as exc:  # noqa: BLE001 - we deliberately retry anything
        warnings.warn(
            f"weights_only=True could not load {path!r} ({exc!r}); "
            "retrying with weights_only=False. Only do this for files you trust.",
            RuntimeWarning,
            stacklevel=2,
        )
        return torch.load(path, map_location=map_location, weights_only=False)


# Kept as an alias so call sites read naturally.
weights_only_load = torch_load


# --------------------------------------------------------------------------
# Automatic mixed precision
# --------------------------------------------------------------------------
def autocast(enabled=True, device_type="cuda"):
    """Return the right autocast context manager for this PyTorch version."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        try:
            return torch.amp.autocast(device_type, enabled=enabled)
        except TypeError:  # very old signature without device_type
            return torch.amp.autocast(enabled=enabled)

    from torch.cuda.amp import autocast as _cuda_autocast  # pragma: no cover

    return _cuda_autocast(enabled=enabled)


def make_grad_scaler(enabled=True, device_type="cuda"):
    """Return a GradScaler, using the non-deprecated namespace when present."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler(device_type, enabled=enabled)
        except (TypeError, ValueError):
            return torch.amp.GradScaler(enabled=enabled)

    from torch.cuda.amp import GradScaler  # pragma: no cover

    return GradScaler(enabled=enabled)


# --------------------------------------------------------------------------
# weight_norm
# --------------------------------------------------------------------------
# ``torch.nn.utils.weight_norm`` is soft-deprecated in favour of
# ``torch.nn.utils.parametrizations.weight_norm``.  They are NOT
# interchangeable: the legacy implementation stores ``weight_g`` / ``weight_v``
# while the new one stores ``parametrizations.weight.original0/1``.  The
# published pretrained checkpoints use the legacy layout, so we must keep using
# the legacy functions for as long as PyTorch still ships them.
_LEGACY_WN = getattr(torch.nn.utils, "weight_norm", None)
_LEGACY_RWN = getattr(torch.nn.utils, "remove_weight_norm", None)


def weight_norm(module, name="weight"):
    """Apply weight normalisation, preferring the legacy (checkpoint-compatible) API."""
    if _LEGACY_WN is not None:
        return _LEGACY_WN(module, name=name)

    # Fallback: parametrization based.  Works for training from scratch but is
    # NOT state-dict compatible with the released pretrained models.
    from torch.nn.utils.parametrizations import weight_norm as _param_wn

    warnings.warn(
        "torch.nn.utils.weight_norm is unavailable; falling back to "
        "torch.nn.utils.parametrizations.weight_norm. Existing pretrained "
        "checkpoints will not load into this model.",
        RuntimeWarning,
        stacklevel=2,
    )
    return _param_wn(module, name=name)


def remove_weight_norm(module, name="weight"):
    """Counterpart of :func:`weight_norm`."""
    if _LEGACY_RWN is not None:
        return _LEGACY_RWN(module, name=name)

    from torch.nn.utils.parametrize import remove_parametrizations

    return remove_parametrizations(module, name)


# --------------------------------------------------------------------------
# Warning hygiene
# --------------------------------------------------------------------------
def silence_known_warnings():
    """Suppress upstream deprecation noise that we intentionally keep using."""
    warnings.filterwarnings(
        "ignore", message=r".*torch\.nn\.utils\.weight_norm.*", category=UserWarning
    )
    warnings.filterwarnings(
        "ignore", message=r".*torch\.nn\.utils\.remove_weight_norm.*", category=UserWarning
    )
    warnings.filterwarnings(
        "ignore", message=r".*`torch\.cuda\.amp.*", category=FutureWarning
    )
    warnings.filterwarnings(
        "ignore", message=r".*return_complex.*", category=UserWarning
    )
    # torchaudio >= 2.9 prints a long notice about TorchCodec on every load().
    warnings.filterwarnings(
        "ignore", message=r".*TorchCodec.*", category=UserWarning
    )
    warnings.filterwarnings(
        "ignore", message=r".*torchaudio\.load.*deprecated.*", category=UserWarning
    )


def project_root():
    """Absolute path of the repository root (the directory holding this file)."""
    return os.path.dirname(os.path.abspath(__file__))
