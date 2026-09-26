"""Monotonic alignment search (MAS) used by the VITS text encoder.

The fast path is the Cython extension built from ``core.pyx``.  Because a
missing C toolchain used to make the whole project unusable, this module now
degrades gracefully:

1. the ``core`` extension  -> built by ``python setup.py build_ext --inplace``
2. legacy nested layout ``monotonic_align.monotonic_align.core`` (old builds)
3. a Numba JIT implementation (Numba ships with librosa, so it is present here)
4. a plain-Python implementation (correct but slow — smoke tests only)

:func:`backend_name` reports which one is active and :func:`backend_error`
explains why the fast paths were unavailable.
"""

import warnings

import numpy as np
import torch

_BACKEND = "unknown"
BACKEND_ERROR = None

try:  # 1) current layout: extension built next to this file
    from .core import maximum_path_c  # type: ignore

    _BACKEND = "cython"
except ImportError:
    try:  # 2) layout produced by the original build instructions
        from .monotonic_align.core import maximum_path_c  # type: ignore

        _BACKEND = "cython (legacy layout)"
    except ImportError:
        maximum_path_c = None


# ---------------------------------------------------------------------------
# Fallback implementations
# ---------------------------------------------------------------------------
def _pure_python_maximum_path_each(path, value, t_y, t_x, max_neg_val=-1e9):
    index = t_x - 1
    for y in range(t_y):
        for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
            if x == y:
                v_cur = max_neg_val
            else:
                v_cur = value[y - 1, x]
            if x == 0:
                v_prev = 0.0 if y == 0 else max_neg_val
            else:
                v_prev = value[y - 1, x - 1]
            value[y, x] += max(v_prev, v_cur)

    for y in range(t_y - 1, -1, -1):
        path[y, index] = 1
        if index != 0 and (index == y or value[y - 1, index] < value[y - 1, index - 1]):
            index = index - 1


def _python_backend(paths, values, t_ys, t_xs):
    for i in range(paths.shape[0]):
        _pure_python_maximum_path_each(paths[i], values[i], int(t_ys[i]), int(t_xs[i]))


def _build_numba_backend():
    """Return ``(callable, error)`` for the Numba backend.

    The kernels are JIT-compiled and exercised once here, so a broken Numba
    install fails at import time instead of in the middle of training.
    """
    try:
        import numba
    except ImportError as exc:
        return None, f"numba is not installed ({exc})"

    def _make(cache):
        @numba.njit(cache=cache)
        def _each(path, value, t_y, t_x, max_neg_val=-1e9):
            index = t_x - 1
            for y in range(t_y):
                for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
                    if x == y:
                        v_cur = max_neg_val
                    else:
                        v_cur = value[y - 1, x]
                    if x == 0:
                        v_prev = 0.0 if y == 0 else max_neg_val
                    else:
                        v_prev = value[y - 1, x - 1]
                    value[y, x] += max(v_prev, v_cur)
            for y in range(t_y - 1, -1, -1):
                path[y, index] = 1
                if index != 0 and (index == y or value[y - 1, index] < value[y - 1, index - 1]):
                    index = index - 1

        @numba.njit(cache=cache, parallel=True, nogil=True)
        def _all(paths, values, t_ys, t_xs):
            for i in numba.prange(paths.shape[0]):
                _each(paths[i], values[i], t_ys[i], t_xs[i])

        return _all

    last_error = None
    # cache=True writes a __pycache__ next to this file, which can fail in
    # read-only or frozen environments — fall back to an uncached build.
    for use_cache in (True, False):
        try:
            backend = _make(use_cache)
            # Warm up with the *same shapes/dtypes* the real call site uses:
            # paths/values are 3-D [b, t_t, t_s], t_ys/t_xs are 1-D int32.
            # Compiling against the wrong rank here would specialise the
            # kernels incorrectly and fail during training.
            probe_path = np.zeros((1, 2, 2), dtype=np.int32)
            probe_value = np.zeros((1, 2, 2), dtype=np.float32)
            backend(probe_path, probe_value,
                    np.array([2], dtype=np.int32), np.array([2], dtype=np.int32))
            return backend, None
        except Exception as exc:  # noqa: BLE001 - any JIT failure -> next option
            last_error = f"{type(exc).__name__}: {exc}"
    return None, last_error


if maximum_path_c is None:
    maximum_path_c, BACKEND_ERROR = _build_numba_backend()
    if maximum_path_c is not None:
        _BACKEND = "numba"
    else:
        _BACKEND = "python (slow)"
        maximum_path_c = _python_backend
        warnings.warn(
            "monotonic_align is running on the pure-Python fallback "
            f"({BACKEND_ERROR}). Training will be extremely slow.\n"
            "Build the Cython extension for full speed:\n"
            "    cd monotonic_align && python setup.py build_ext --inplace",
            RuntimeWarning,
            stacklevel=2,
        )


def backend_name():
    """Name of the active MAS backend (for logs / diagnostics)."""
    return _BACKEND


def backend_error():
    """Why the fast backends were unavailable, or None."""
    return BACKEND_ERROR


def maximum_path(neg_cent, mask):
    """Monotonic alignment search.

    neg_cent: [b, t_t, t_s]
    mask:     [b, t_t, t_s]
    """
    device = neg_cent.device
    dtype = neg_cent.dtype
    neg_cent = np.ascontiguousarray(neg_cent.detach().cpu().numpy(), dtype=np.float32)
    path = np.zeros(neg_cent.shape, dtype=np.int32)

    t_t_max = np.ascontiguousarray(
        mask.sum(1)[:, 0].detach().cpu().numpy(), dtype=np.int32
    )
    t_s_max = np.ascontiguousarray(
        mask.sum(2)[:, 0].detach().cpu().numpy(), dtype=np.int32
    )
    maximum_path_c(path, neg_cent, t_t_max, t_s_max)
    return torch.from_numpy(path).to(device=device, dtype=dtype)
