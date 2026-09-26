import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_io import load_audio, save_audio  # noqa: E402


def main():
    with open("./configs/finetune_speaker.json", 'r', encoding='utf-8') as f:
        hps = json.load(f)
    target_sr = hps['data']['sampling_rate']
    source_dir = "./sampled_audio4ft"
    if not os.path.isdir(source_dir):
        print(f"{source_dir} not found — the auxiliary dataset was not downloaded, "
              "skipping resampling.")
        return
    filelist = list(os.walk(source_dir))[0][2]
    if target_sr != 22050:
        for wavfile in filelist:
            wav, sr = load_audio(os.path.join(source_dir, wavfile), target_sr=target_sr, mono=True)
            save_audio(os.path.join(source_dir, wavfile), wav, target_sr)
        print(f"Resampled {len(filelist)} auxiliary files to {target_sr} Hz.")
    else:
        print("Target sampling rate is 22050 Hz — no resampling needed.")


if __name__ == "__main__":
    main()
