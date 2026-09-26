"""Vocal separation + resampling.

Uses the Demucs CLI to strip background music/noise, then resamples everything
to the model's sampling rate.

Audio IO goes through ``audio_io`` (soundfile) instead of ``torchaudio.load``:
torchaudio's I/O layer moved to TorchCodec in 2.9, where the old
``normalize=`` argument is ignored and loading fails outright if TorchCodec
isn't installed.
"""

import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_io import load_audio, save_audio  # noqa: E402

raw_audio_dir = "./raw_audio/"
denoise_audio_dir = "./denoised_audio/"
separated_dir = "./separated/"


def _have_demucs():
    return shutil.which("demucs") is not None


def main():
    os.makedirs(denoise_audio_dir, exist_ok=True)

    # 2023/4/21: Get the target sampling rate
    with open("./configs/finetune_speaker.json", 'r', encoding='utf-8') as f:
        hps = json.load(f)
    target_sr = hps['data']['sampling_rate']

    if not os.path.isdir(raw_audio_dir):
        print(f"{raw_audio_dir} does not exist — nothing to do.")
        return
    filelist = list(os.walk(raw_audio_dir))[0][2]
    wav_files = [f for f in filelist if f.endswith(".wav")]

    if not wav_files:
        print(f"No .wav files found under {raw_audio_dir} — nothing to do.")
        return

    if _have_demucs():
        for file in wav_files:
            subprocess.run(
                ["demucs", "--two-stems=vocals", f"{raw_audio_dir}{file}"],
                check=False,
            )
    else:
        print(
            "WARNING: the `demucs` command was not found on PATH, so vocal "
            "separation is being skipped.\n"
            "         Install it with:  pip install demucs\n"
            "         Continuing with the raw audio instead."
        )

    separated_ok = os.path.isdir(os.path.join(separated_dir, "htdemucs"))

    for file in filelist:
        stem = os.path.splitext(file)[0]
        vocals = os.path.join(separated_dir, "htdemucs", stem, "vocals.wav")

        if separated_ok and os.path.isfile(vocals):
            source = vocals
        else:
            if not file.endswith(".wav"):
                continue
            source = os.path.join(raw_audio_dir, file)

        wav, sr = load_audio(source, target_sr=target_sr, mono=True)
        save_audio(os.path.join(denoise_audio_dir, stem + ".wav"), wav, target_sr)

    print(f"Done — denoised audio written to {denoise_audio_dir}")


if __name__ == "__main__":
    main()
