import librosa
import numpy as np
import torch
from librosa.filters import mel as librosa_mel_fn

MAX_WAV_VALUE = 32768.0


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def _stft_magnitude(y, n_fft, hop_size, win_size, center, window):
    """Magnitude spectrogram that works on every supported torch version.

    ``torch.stft(..., return_complex=False)`` has been deprecated since 2.x, so
    we ask for a complex result and split it back into real/imaginary with
    ``torch.view_as_real`` — mathematically identical to the old behaviour.
    """
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.view_as_real(spec)  # [..., freq, 2]
    return torch.sqrt(spec.pow(2).sum(-1) + 1e-6)


def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    wnsize_dtype_device = str(win_size) + "_" + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(
            dtype=y.dtype, device=y.device
        )

    y = torch.nn.functional.pad(
        y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode="reflect"
    )
    y = y.squeeze(1)

    return _stft_magnitude(y, n_fft, hop_size, win_size, center, hann_window[wnsize_dtype_device])


def spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax):
    global mel_basis
    dtype_device = str(spec.dtype) + "_" + str(spec.device)
    fmax_dtype_device = str(fmax) + "_" + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel_basis[fmax_dtype_device] = _mel_basis(sampling_rate, n_fft, num_mels, fmin, fmax, spec)
    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    spec = spectral_normalize_torch(spec)
    return spec


def mel_spectrogram_torch(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global mel_basis, hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    fmax_dtype_device = str(fmax) + "_" + dtype_device
    wnsize_dtype_device = str(win_size) + "_" + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel_basis[fmax_dtype_device] = _mel_basis(sampling_rate, n_fft, num_mels, fmin, fmax, y)
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(
            dtype=y.dtype, device=y.device
        )

    y = torch.nn.functional.pad(
        y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode="reflect"
    )
    y = y.squeeze(1)

    spec = _stft_magnitude(y.float(), n_fft, hop_size, win_size, center, hann_window[wnsize_dtype_device])
    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    spec = spectral_normalize_torch(spec)

    return spec


def _mel_basis(sampling_rate, n_fft, num_mels, fmin, fmax, ref_tensor):
    """Build the mel filter bank.

    librosa made every argument of ``filters.mel`` keyword-only in 0.10, so the
    old positional call ``mel(sr, n_fft, n_mels, fmin, fmax)`` now raises
    ``TypeError``.  Always pass by keyword.
    """
    mel = librosa_mel_fn(
        sr=int(sampling_rate),
        n_fft=int(n_fft),
        n_mels=int(num_mels),
        fmin=float(fmin),
        fmax=None if fmax is None else float(fmax),
    )
    return torch.from_numpy(np.ascontiguousarray(mel)).to(
        dtype=ref_tensor.dtype, device=ref_tensor.device
    )
