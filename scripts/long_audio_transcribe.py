import json
import os
import re
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
import whisper  # noqa: E402

from audio_io import load_audio, save_audio  # noqa: E402

parent_dir = "./denoised_audio/"

AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".opus"}

# {CharacterName}_{number}, e.g. Taffy_332452. Must stay in sync with
# scripts/organize_data.py, which normalises file names into this shape.
_NUMBERED = re.compile(r"^(?P<name>.+)_(?P<code>\d+)$")


def split_character(stem):
    """``'Taffy_332452'`` -> ``('Taffy', '332452')``; ``'myvoice'`` -> ``('myvoice', '0')``.

    The old code used ``file.rstrip(".wav").split("_")[1]``, which crashed on a
    name without an underscore and mangled stems ending in w/a/v (``rstrip``
    treats its argument as a *character set*, so ``"aaa.wav"`` -> ``""``).
    """
    m = _NUMBERED.match(stem)
    if m:
        return m.group("name"), m.group("code")
    return stem, "0"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", default="CJE")
    parser.add_argument("--whisper_size", default="medium")
    args = parser.parse_args()
    if args.languages == "CJE":
        lang2token = {
            'zh': "[ZH]",
            'ja': "[JA]",
            "en": "[EN]",
        }
    elif args.languages == "CJ":
        lang2token = {
            'zh': "[ZH]",
            'ja': "[JA]",
        }
    elif args.languages == "C":
        lang2token = {
            'zh': "[ZH]",
        }
    else:
        raise ValueError(f"Unknown --languages {args.languages!r}; expected CJE / CJ / C")

    if not torch.cuda.is_available():
        print("WARNING: no GPU detected. Whisper will run on CPU and will be slow; "
              "consider --whisper_size small or medium.")

    with open("./configs/finetune_speaker.json", 'r', encoding='utf-8') as f:
        hps = json.load(f)
    target_sr = hps['data']['sampling_rate']
    os.makedirs("./segmented_character_voice", exist_ok=True)

    model = None
    if not os.path.isdir(parent_dir):
        print(f"{parent_dir} 不存在 —— 上一步 denoise_audio.py 没有产出？")
        filelist = []
    else:
        filelist = sorted(f for f in os.listdir(parent_dir)
                          if os.path.isfile(os.path.join(parent_dir, f))
                          and os.path.splitext(f)[1].lower() in AUDIO_EXT)
        if not filelist:
            print(f"{parent_dir} 里没有音频文件。")
            print("这属于正常情况：如果你只上传了短音频（custom_character_voice/），"
                  "就没有长音频需要切分。")
            print("如果确实上传了长音频/视频，请确认：")
            print("  1) 文件已放进 ./raw_audio/ 或 ./video_data/")
            print("  2) 文件名形如 <角色名>_<数字>.wav"
                  "（可运行 scripts/organize_data.py 自动补齐）")

    if filelist:
        model = whisper.load_model(args.whisper_size)

    speaker_annos = []
    unnamed = []
    for file in filelist:
        audio_path = os.path.join(parent_dir, file)
        stem = os.path.splitext(file)[0]
        character_name, code = split_character(stem)
        if not _NUMBERED.match(stem):
            unnamed.append(file)

        print(f"Transcribing {audio_path}...")

        options = dict(beam_size=5, best_of=5)
        transcribe_options = dict(task="transcribe", **options)

        result = model.transcribe(audio_path, word_timestamps=True, **transcribe_options)
        segments = result["segments"]
        lang = result['language']
        if lang not in lang2token:
            print(f"  {lang} not supported, ignoring...")
            continue

        outdir = os.path.join("./segmented_character_voice", character_name)
        os.makedirs(outdir, exist_ok=True)

        wav, sr = load_audio(audio_path, mono=True)

        for i, seg in enumerate(segments):
            start_time = seg['start']
            end_time = seg['end']
            text = seg['text']
            text_tokened = lang2token[lang] + text.replace("\n", "") + lang2token[lang] + "\n"
            start_idx = int(start_time * sr)
            end_idx = int(end_time * sr)
            num_samples = end_idx - start_idx
            if num_samples <= 0:
                print(f"Skipping zero-length segment: start={start_time}, end={end_time}")
                continue
            wav_seg = wav[:, start_idx:end_idx]
            if wav_seg.shape[1] == 0:
                print(f"Skipping empty segment i={i}, shape={wav_seg.shape}")
                continue
            if sr != target_sr:
                from audio_io import resample as _resample
                wav_seg = _resample(wav_seg, sr, target_sr)

            wav_seg_name = f"{character_name}_{code}_{i}.wav"
            savepth = os.path.join(outdir, wav_seg_name)
            speaker_annos.append(savepth + "|" + character_name + "|" + text_tokened)
            save_audio(savepth, wav_seg, target_sr)

        print(f"  -> {character_name}: {len(segments)} 个片段")

    if unnamed:
        print(f"\n⚠️ {len(unnamed)} 个文件名不含 '_<数字>'，已把整个文件名当作角色名。")
        print("   DATA.MD 推荐的命名是 <角色名>_<数字>.wav，例如 Diana_234135.wav。")
        print("   可以运行  python scripts/organize_data.py  自动补齐。")

    if len(speaker_annos) == 0:
        print("Warning: no long audios & videos found, this IS expected if you have only uploaded short audios")
        print("this IS NOT expected if you have uploaded any long audios, videos or video links. Please check your file structure or make sure your audio/video language is supported.")
    else:
        print(f"\n✅ 共切分标注 {len(speaker_annos)} 条 -> long_character_anno.txt")

    with open("./long_character_anno.txt", 'w', encoding='utf-8') as f:
        for line in speaker_annos:
            f.write(line)
