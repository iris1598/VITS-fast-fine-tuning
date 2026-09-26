import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
import whisper  # noqa: E402

from audio_io import load_audio, save_audio  # noqa: E402

lang2token = {
    'zh': "[ZH]",
    'ja': "[JA]",
    "en": "[EN]",
}

AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".opus"}

# Whisper always works on a 30 second window (`pad_or_trim`). Clips up to this
# length are transcribed in full; longer ones would silently lose content, so
# they are reported instead of being fed in half-empty.
WHISPER_WINDOW = 30.0


def transcribe_one(model, audio_path):
    """Transcribe a <=30s clip with Whisper."""
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    lang = max(probs, key=probs.get)

    # decode the audio
    options = whisper.DecodingOptions(beam_size=5)
    result = whisper.decode(model, mel, options)

    return lang, result.text


def list_speaker_dirs(parent_dir):
    """Immediate sub-directories of ``parent_dir`` that actually hold audio."""
    if not os.path.isdir(parent_dir):
        return []
    out = []
    for name in sorted(os.listdir(parent_dir)):
        path = os.path.join(parent_dir, name)
        if not os.path.isdir(path) or name.startswith("."):
            continue
        audio = [f for f in sorted(os.listdir(path))
                 if os.path.splitext(f)[1].lower() in AUDIO_EXT]
        if audio:
            out.append((name, audio))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", default="CJE")
    parser.add_argument("--whisper_size", default="medium")
    args = parser.parse_args()
    if args.languages == "CJE":
        lang2token = {'zh': "[ZH]", 'ja': "[JA]", "en": "[EN]"}
    elif args.languages == "CJ":
        lang2token = {'zh': "[ZH]", 'ja': "[JA]"}
    elif args.languages == "C":
        lang2token = {'zh': "[ZH]"}
    if not torch.cuda.is_available():
        print("WARNING: no GPU detected. Whisper will run on CPU and will be slow; "
              "consider --whisper_size small or medium.")
    model = whisper.load_model(args.whisper_size)

    parent_dir = "./custom_character_voice/"
    speakers = list_speaker_dirs(parent_dir)

    with open("./configs/finetune_speaker.json", 'r', encoding='utf-8') as f:
        hps = json.load(f)
    target_sr = hps['data']['sampling_rate']

    speaker_annos = []
    skipped_long = []
    failed = 0

    if not speakers:
        print(f"{parent_dir} 下没有找到「角色名/音频」的结构。")
        print("请先运行：  python scripts/organize_data.py --character <角色名>")
    else:
        total = sum(len(files) for _, files in speakers)
        print(f"找到 {len(speakers)} 个角色，共 {total} 个音频文件\n")

    processed = 0
    for speaker, wavfiles in speakers:
        outdir_ok = True
        for i, wavfile in enumerate(wavfiles):
            if wavfile.startswith("processed_"):
                continue
            src = os.path.join(parent_dir, speaker, wavfile)
            try:
                wav, sr = load_audio(src, mono=True)
                if sr != target_sr:
                    from audio_io import resample as _resample
                    wav = _resample(wav, sr, target_sr)

                duration = wav.shape[1] / target_sr
                if duration > WHISPER_WINDOW:
                    skipped_long.append((speaker, wavfile, duration))
                    continue

                os.makedirs(os.path.join(parent_dir, speaker), exist_ok=True)
                save_path = os.path.join(parent_dir, speaker, f"processed_{processed}.wav")
                save_audio(save_path, wav, target_sr)

                lang, text = transcribe_one(model, save_path)
                if lang not in list(lang2token.keys()):
                    print(f"  {speaker}/{wavfile}: 语言 {lang} 不受支持，跳过")
                    continue
                text = lang2token[lang] + text + lang2token[lang] + "\n"
                speaker_annos.append(save_path + "|" + speaker + "|" + text)

                processed += 1
                print(f"  [{processed}] {speaker}/{wavfile} ({duration:.1f}s, {lang}) "
                      f"-> {text.strip()[:48]}")
            except Exception as exc:  # noqa: BLE001 - keep going through the batch
                failed += 1
                print(f"  ! {speaker}/{wavfile}: {exc!r}")

    # --- report -----------------------------------------------------------
    print()
    if skipped_long:
        print(f"⚠️ {len(skipped_long)} 个文件超过 {WHISPER_WINDOW:.0f} 秒，未处理：")
        for speaker, name, dur in skipped_long[:10]:
            print(f"    {speaker}/{name}  ({dur:.1f}s)")
        if len(skipped_long) > 10:
            print(f"    ... 其余 {len(skipped_long) - 10} 个")
        print("  这些放进 custom_character_voice/ 会被 Whisper 截断、且转写文本超过")
        print("  preprocess_v2.py 的 150 字上限而被丢弃。如需使用，请改放到 raw_audio/：")
        print("    mv ./custom_character_voice/<角色>/<文件> ./raw_audio/<角色>_0.wav")
        print("  然后让 STEP 3 的 long_audio_transcribe.py 去自动切分标注。")
    if failed:
        print(f"⚠️ {failed} 个文件读取/转写失败（已在上面逐条列出原因）")

    if len(speaker_annos) == 0:
        print("Warning: no short audios found, this IS expected if you have only uploaded long audios, videos or video links.")
        print("this IS NOT expected if you have uploaded a zip file of short audios. Please check your file structure or make sure your audio language is supported.")
    else:
        print(f"✅ 共标注 {len(speaker_annos)} 条短音频 -> short_character_anno.txt")

    with open("short_character_anno.txt", 'w', encoding='utf-8') as f:
        for line in speaker_annos:
            f.write(line)
