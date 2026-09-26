"""Stable audio I/O used by every stage of the pipeline.

Historically this project called ``torchaudio.load`` / ``torchaudio.save``
directly.  torchaudio's I/O layer has since been re-based on TorchCodec
(2.9+) and the ``normalize`` / ``backend`` / ``buffer_size`` arguments are now
ignored or, worse, ``torchaudio.load`` raises ``ImportError`` when TorchCodec
is not installed.  Rather than depend on that churn we use ``soundfile`` as the
primary backend (it is what librosa uses under the hood anyway) and keep
torchaudio / librosa as fallbacks.

Public API — all tensors are float32:
    ``load_audio(path, target_sr=None)`` -> ``(FloatTensor[C, T], sample_rate)``
    ``save_audio(path, wav, sample_rate)``
    ``resample(wav, orig_sr, new_sr)`` -> ``FloatTensor[C, T]``
    ``to_mono(wav)`` -> ``FloatTensor[1, T]``
"""

import warnings

import numpy as np
import torch

__all__ = ["load_audio", "save_audio", "resample", "to_mono", "peak_normalize"]

try:  # primary backend
    import soundfile as _sf
except ImportError:  # pragma: no cover
    _sf = None

try:
    import torchaudio as _ta
except ImportError:  # pragma: no cover
    _ta = None


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def _load_soundfile(path):
    data, sr = _sf.read(str(path), dtype="float32", always_2d=True)
    # soundfile is samples-first: [T, C] -> torch wants [C, T]
    return torch.from_numpy(np.ascontiguousarray(data.T)), int(sr)


def _load_torchaudio(path):
    wav, sr = _ta.load(str(path), channels_first=True)
    if not torch.is_floating_point(wav):
        # normalize=False style integer PCM -> map to [-1, 1)
        info = torch.iinfo(wav.dtype)
        wav = wav.to(torch.float32) / float(max(abs(info.min), info.max))
    return wav.to(torch.float32), int(sr)


def _load_librosa(path):
    import librosa

    data, sr = librosa.load(str(path), sr=None, mono=False)
    if data.ndim == 1:
        data = data[None, :]
    return torch.from_numpy(np.ascontiguousarray(data)).float(), int(sr)


def load_audio(path, target_sr=None, mono=True):
    """Load an audio file as a float32 ``[C, T]`` tensor.

    Args:
        path: file path.
        target_sr: if given, resample to this rate.
        mono: average channels down to a single channel.
    """
    errors = []
    wav = sr = None

    loaders = []
    if _sf is not None:
        loaders.append(_load_soundfile)
    if _ta is not None:
        loaders.append(_load_torchaudio)
    loaders.append(_load_librosa)

    for loader in loaders:
        try:
            wav, sr = loader(path)
            break
        except Exception as exc:  # noqa: BLE001 - try the next backend
            errors.append(f"{loader.__name__}: {exc}")

    if wav is None:
        raise RuntimeError(
            "Could not decode audio file {!r}. Tried:\n  {}".format(
                str(path), "\n  ".join(errors)
            )
        )

    if mono:
        wav = to_mono(wav)
    if target_sr is not None and sr != target_sr:
        wav = resample(wav, sr, target_sr)
        sr = int(target_sr)

    return wav.contiguous(), sr


def to_mono(wav):
    """Down-mix ``[C, T]`` (or ``[T]``) to a single channel."""
    if wav.dim() == 1:
        return wav.unsqueeze(0)
    if wav.size(0) == 1:
        return wav
    return wav.mean(dim=0, keepdim=True)


# --------------------------------------------------------------------------
# Resampling
# --------------------------------------------------------------------------
def resample(wav, orig_sr, new_sr):
    """Resample ``[C, T]`` float32 audio."""
    orig_sr, new_sr = int(orig_sr), int(new_sr)
    if orig_sr == new_sr:
        return wav

    if _ta is not None:
        # torchaudio.transforms.Resample is pure tensor math and has kept a
        # stable signature across every 2.x release.
        return _ta.transforms.Resample(orig_freq=orig_sr, new_freq=new_sr)(wav.float())

    import librosa  # pragma: no cover

    out = librosa.resample(
        wav.detach().cpu().numpy(),
        orig_sr=orig_sr,
        target_sr=new_sr,
        axis=-1,
    )
    return torch.from_numpy(np.ascontiguousarray(out)).float()  # pragma: no cover


# --------------------------------------------------------------------------
# Saving
# --------------------------------------------------------------------------
def save_audio(path, wav, sample_rate):
    """Write a float32 ``[C, T]`` (or ``[T]``) tensor to ``path``."""
    if isinstance(wav, torch.Tensor):
        wav = wav.detach().cpu().float().numpy()
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim == 2:
        wav = wav.T  # back to samples-first for soundfile

    if _sf is not None:
        _sf.write(str(path), wav, int(sample_rate))
        return

    if _ta is not None:  # pragma: no cover
        tensor = torch.from_numpy(np.ascontiguousarray(wav.T if wav.ndim == 2 else wav))
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        _ta.save(str(path), tensor, int(sample_rate), channels_first=True)
        return

    raise RuntimeError("No audio backend available to write WAV files.")


def peak_normalize(wav, eps=1e-8):
    """Scale a tensor so its peak magnitude is just below 1.0 (kept for parity
    with the original voice-conversion preprocessing)."""
    scale = max(-wav.min().item(), wav.max().item())
    if scale < eps:
        return wav
    return wav / scale / 0.99
