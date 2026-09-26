"""Environment self-check for VITS-fast-fine-tuning.

Run this after installing the dependencies to confirm that everything the
training / inference pipeline needs is actually working on your machine:

    python scripts/verify_install.py

It checks, in order:
  1. Python / PyTorch / NumPy versions
  2. importability of every project module
  3. the monotonic alignment backend (Cython > Numba > Python fallback)
  4. the text frontends required by a config's `data.text_cleaners`
  5. audio I/O (soundfile) round-trip
  6. the dataset + collate pipeline on a generated clip
  7. one complete training step (generator + discriminator, forward + backward)
  8. model inference

Exit code 0 means the environment is good.
"""

import argparse
import os
import sys
import tempfile
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"
_results = []


class SoftFail(Exception):
    """Raised when something is missing but is not required to keep going
    (e.g. an optional language backend for a model you may not be using)."""


def check(name, fn):
    try:
        detail = fn()
        _results.append((PASS, name, detail or ""))
        print(f"[{PASS}] {name}" + (f" — {detail}" if detail else ""))
        return True
    except SoftFail as exc:
        _results.append((WARN, name, str(exc)))
        print(f"[{WARN}] {name} — {exc}")
        return True
    except Exception as exc:  # noqa: BLE001 - this is a diagnostic tool
        _results.append((FAIL, name, repr(exc)))
        print(f"[{FAIL}] {name} — {exc!r}")
        if os.environ.get("VERIFY_TRACEBACK"):
            traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
def step_versions():
    import numpy
    import torch

    cuda = torch.cuda.is_available()
    detail = (f"python={sys.version.split()[0]} torch={torch.__version__} "
              f"numpy={numpy.__version__} cuda={cuda}")
    if not cuda:
        detail += "  (no GPU: training will be very slow, smoke tests only)"
    return detail


def step_imports():
    import importlib

    modules = [
        "compat", "audio_io", "commons", "mel_processing", "modules", "models",
        "attentions", "transforms", "losses", "data_utils", "utils",
        "text", "text.cleaners",
    ]
    failed = []
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{name}: {exc!r}")
    if failed:
        raise RuntimeError("; ".join(failed))
    # scripts are imported implicitly by running them; just check they compile
    for script in os.listdir(os.path.join(ROOT, "scripts")):
        if script.endswith(".py"):
            path = os.path.join(ROOT, "scripts", script)
            with open(path, encoding="utf-8") as f:
                compile(f.read(), path, "exec")
    return f"{len(modules)} modules + scripts/ compile"


def step_monotonic_align():
    import monotonic_align

    backend = monotonic_align.backend_name()
    if backend.startswith("python"):
        raise RuntimeError(
            "only the slow fallback is available "
            f"({monotonic_align.backend_error()}). Build it with: "
            "cd monotonic_align && python setup.py build_ext --inplace"
        )

    import numpy as np
    import torch

    rng = np.random.RandomState(0)
    neg = torch.tensor(rng.randn(3, 9, 14), dtype=torch.float32)
    mask = torch.zeros(3, 9, 14)
    for i in range(3):
        mask[i, : 9 - i, : 14 - i] = 1
    path = monotonic_align.maximum_path(neg, mask)

    for b in range(3):
        used_cols = []
        for row in range(path.shape[1]):
            cols = torch.nonzero(path[b, row]).flatten().tolist()
            if cols:
                used_cols.append(cols[0])
        if used_cols != sorted(used_cols):
            raise RuntimeError(f"alignment for batch {b} is not monotonic: {used_cols}")
    return f"backend={backend}"


def step_text_backends():
    """Each language frontend must actually produce phonemes."""
    import importlib

    probes = [
        ("mandarin", "text.mandarin", "chinese_to_ipa", "今天天气不错"),
        ("english", "text.english", "english_to_ipa2", "hello world"),
        ("korean", "text.korean", "korean_to_ipa", "안녕하세요"),
        ("japanese", "text.japanese", "japanese_to_ipa2", "こんにちは"),
    ]
    ok, missing = [], []
    for lang, module_name, func_name, text in probes:
        try:
            mod = importlib.import_module(module_name)
        except ImportError:
            missing.append(module_name)
            continue
        out = getattr(mod, func_name)(text)
        if not out or len(out.strip()) < 2:
            raise RuntimeError(f"{lang} frontend returned {out!r} for {text!r}")
        ok.append(lang)

    detail = "available: " + ", ".join(ok)
    if missing:
        # Japanese is the only commonly-required optional one; report it but
        # don't fail, since a Chinese-only model does not need it.
        detail += " | not installed: " + ", ".join(missing)
        raise SoftFail(detail)
    return detail


def step_text_frontend(config_path):
    """The cleaners named by the config must run and match its symbol table."""
    import json

    with open(config_path, encoding="utf-8") as f:
        hps = json.load(f)
    import text

    cleaner_names = hps["data"]["text_cleaners"]

    # Unknown cleaner names are always a hard failure — they mean the config is
    # wrong, not that an optional package is missing.
    unknown = [n for n in cleaner_names if not hasattr(text.cleaners, n)]
    if unknown:
        raise RuntimeError(f"config refers to unknown cleaner(s): {unknown}")

    missing = text.check_cleaners(cleaner_names)
    if missing:
        raise SoftFail(
            "configured cleaner(s) need a backend that is not installed: "
            + "; ".join(missing)
        )

    probe = "[ZH]今天天气不错。[JA]こんにちは。[EN]hello world."
    cleaned = text._clean_text(probe, cleaner_names)
    seq = text.cleaned_text_to_sequence(cleaned, hps["symbols"])

    if not cleaned.strip():
        raise RuntimeError("cleaner produced an empty string")
    if not seq:
        raise RuntimeError(
            "none of the cleaned text could be mapped to the config's symbol "
            f"table. cleaned={cleaned[:60]!r}"
        )

    # A low coverage ratio means the config's `symbols` were generated for a
    # different cleaner. This happens with the stale sample configs shipped in
    # the repo — the real config is downloaded in STEP 1.5.
    coverage = len(seq) / max(len(cleaned), 1)
    if coverage < 0.5:
        raise SoftFail(
            f"only {coverage:.0%} of the cleaned text maps to config symbols — "
            f"the symbol table in {os.path.basename(config_path)} does not match "
            f"{cleaner_names}. Regenerate it (STEP 3.5 writes "
            "configs/modified_finetune_speaker.json) or use the downloaded config."
        )
    return f"cleaners={cleaner_names} coverage={coverage:.0%} -> {len(seq)} tokens"


def step_lazy_backends():
    """A missing optional language package must not break `import text`."""
    import importlib

    import text
    importlib.reload(text)

    from text import cleaners

    status = []
    for name, module in (("japanese", "pyopenjtalk"),
                         ("korean", "jamo"),
                         ("mandarin", "pypinyin"),
                         ("english", "eng_to_ipa")):
        try:
            importlib.import_module(module)
            status.append(f"{name}=ok")
        except ImportError:
            status.append(f"{name}=missing(lazy)")
    # this must not raise
    cleaners.chinese_cleaners("测试")
    return ", ".join(status)


def step_audio_io():
    import numpy as np
    import torch

    from audio_io import load_audio, resample, save_audio

    sr = 22050
    t = np.arange(sr, dtype=np.float32) / sr
    wav = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "tone.wav")
        save_audio(path, torch.from_numpy(wav).unsqueeze(0), sr)
        loaded, got_sr = load_audio(path)
        assert got_sr == sr, f"sample rate mismatch {got_sr} != {sr}"
        assert loaded.shape == (1, sr), f"shape mismatch {tuple(loaded.shape)}"
        assert loaded.dtype == torch.float32, loaded.dtype
        assert abs(loaded.abs().max().item() - 0.5) < 0.01, loaded.abs().max().item()

        rs = resample(loaded, sr, 16000)
        assert abs(rs.shape[-1] - 16000) <= 2, f"resampled length {rs.shape[-1]}"
    return "soundfile round-trip + resample"


def step_train_step(config_path):
    """Run exactly the maths finetune_speaker_v2.py runs, once, on synthetic data."""
    import json

    import torch
    from torch.nn import functional as F

    import commons
    from compat import autocast, make_grad_scaler
    from data_utils import TextAudioSpeakerCollate, TextAudioSpeakerLoader
    from losses import discriminator_loss, feature_loss, generator_loss, kl_loss
    from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
    from models import MultiPeriodDiscriminator, SynthesizerTrn
    from utils import HParams

    with open(config_path, encoding="utf-8") as f:
        raw = json.load(f)
    # Keep the smoke test cheap and independent of the real corpus.
    raw["data"]["n_speakers"] = 2
    raw["train"]["batch_size"] = 2
    hps = HParams(**raw)
    hps.data.max_text_len = 400

    with tempfile.TemporaryDirectory() as tmp:
        # --- synthesize a tiny corpus -------------------------------------
        import numpy as np

        from audio_io import save_audio

        sr = hps.data.sampling_rate
        rng = np.random.RandomState(0)
        annos = []
        for i in range(4):
            seconds = 1.0 + 0.25 * i
            n = int(sr * seconds)
            t = np.arange(n, dtype=np.float32) / sr
            wav = 0.3 * np.sin(2 * np.pi * (180 + 40 * i) * t)
            path = os.path.join(tmp, f"spk{i % 2}_{i}.wav")
            save_audio(path, torch.from_numpy(wav.astype(np.float32)).unsqueeze(0), sr)
            annos.append(f"{path}|{i % 2}|ababababa\n")

        anno_path = os.path.join(tmp, "anno.txt")
        with open(anno_path, "w", encoding="utf-8") as f:
            f.writelines(annos)

        hps.data.training_files = anno_path
        hps.data.validation_files = anno_path
        symbols = hps.symbols

        dataset = TextAudioSpeakerLoader(anno_path, hps.data, symbols)
        if len(dataset) == 0:
            raise RuntimeError("dataset ended up empty — annotation/length filtering is broken")
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=2, shuffle=False,
            collate_fn=TextAudioSpeakerCollate())

        net_g = SynthesizerTrn(
            len(symbols), hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers, **hps.model)
        net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm)
        optim_g = torch.optim.AdamW(net_g.parameters(), hps.train.learning_rate,
                                    betas=hps.train.betas, eps=hps.train.eps)
        optim_d = torch.optim.AdamW(net_d.parameters(), hps.train.learning_rate,
                                    betas=hps.train.betas, eps=hps.train.eps)
        scaler = make_grad_scaler(enabled=False, device_type="cpu")

        x, x_lengths, spec, spec_lengths, y, y_lengths, speakers = next(iter(loader))
        with autocast(enabled=False, device_type="cpu"):
            (y_hat, l_length, attn, ids_slice, x_mask, z_mask,
             (z, z_p, m_p, logs_p, m_q, logs_q)) = net_g(
                x, x_lengths, spec, spec_lengths, speakers)
            mel = spec_to_mel_torch(spec, hps.data.filter_length, hps.data.n_mel_channels,
                                    hps.data.sampling_rate, hps.data.mel_fmin, hps.data.mel_fmax)
            y_mel = commons.slice_segments(mel, ids_slice,
                                           hps.train.segment_size // hps.data.hop_length)
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1), hps.data.filter_length, hps.data.n_mel_channels,
                hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
                hps.data.mel_fmin, hps.data.mel_fmax)
            y_seg = commons.slice_segments(y, ids_slice * hps.data.hop_length,
                                           hps.train.segment_size)
            assert torch.isfinite(y_hat).all(), "generator produced non-finite values"

            y_d_r, y_d_g, _, _ = net_d(y_seg, y_hat.detach())
            loss_disc, _, _ = discriminator_loss(y_d_r, y_d_g)

        optim_d.zero_grad()
        loss_disc.backward()
        optim_d.step()

        with autocast(enabled=False, device_type="cpu"):
            y_d_r, y_d_g, fmap_r, fmap_g = net_d(y_seg, y_hat)
            loss_dur = torch.sum(l_length.float())
            loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
            loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
            loss_fm = feature_loss(fmap_r, fmap_g)
            loss_gen, _ = generator_loss(y_d_g)
            loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl

        optim_g.zero_grad()
        loss_gen_all.backward()
        optim_g.step()

        # gradient actually reached the speaker embedding
        assert net_g.emb_g.weight.grad is not None, "no gradient on emb_g"

        return (f"loss_d={loss_disc.item():.3f} loss_g={loss_gen_all.item():.3f} "
                f"batch={tuple(x.shape)}")


def step_inference(config_path):
    import json

    import torch

    from models import SynthesizerTrn
    from utils import HParams

    with open(config_path, encoding="utf-8") as f:
        raw = json.load(f)
    raw["data"]["n_speakers"] = 2
    hps = HParams(**raw)

    net_g = SynthesizerTrn(
        len(hps.symbols), hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers, **hps.model)
    net_g.eval()

    x = torch.randint(1, len(hps.symbols), (1, 12), dtype=torch.long)
    x_lengths = torch.LongTensor([12])
    sid = torch.LongTensor([0])
    with torch.no_grad():
        audio = net_g.infer(x, x_lengths, sid=sid,
                            noise_scale=0.667, noise_scale_w=0.8, length_scale=1.0)[0]
    assert audio.dim() == 3 and audio.shape[1] == 1, tuple(audio.shape)
    assert torch.isfinite(audio).all()
    return f"audio shape={tuple(audio.shape)}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(ROOT, "configs/uma_trilingual.json"),
                        help="config used for the model / text checks")
    args = parser.parse_args()

    print("=" * 74)
    print("VITS-fast-fine-tuning — environment check")
    print("=" * 74)

    check("versions", step_versions)
    check("module imports", step_imports)
    check("monotonic_align", step_monotonic_align)
    check("lazy optional backends", step_lazy_backends)
    check("text language backends", step_text_backends)
    check("config text frontend", lambda: step_text_frontend(args.config))
    check("audio I/O", step_audio_io)
    check("data pipeline + training step", lambda: step_train_step(args.config))
    check("inference", lambda: step_inference(args.config))

    print("=" * 74)
    failed = [r for r in _results if r[0] == FAIL]
    warned = [r for r in _results if r[0] == WARN]
    for status, name, detail in warned:
        print(f"[WARN] {name}: {detail}")
    if failed:
        print(f"{len(failed)} check(s) FAILED:")
        for status, name, detail in failed:
            print(f"  - {name}: {detail}")
        print("\nSet VERIFY_TRACEBACK=1 to print full tracebacks.")
        return 1
    if warned:
        print(f"\nAll hard checks passed ({len(warned)} warning(s) above).")
    else:
        print("All checks passed — you are ready to train.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
