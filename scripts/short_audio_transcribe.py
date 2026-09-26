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
    print(f"Detected language: {lang}")

    # decode the audio
    options = whisper.DecodingOptions(beam_size=5)
    result = whisper.decode(model, mel, options)

    print(result.text)
    return lang, result.text


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
    if not os.path.isdir(parent_dir):
        print(f"{parent_dir} does not exist — did you upload a zip of short audios?")
        speaker_names = []
    else:
        speaker_names = list(os.walk(parent_dir))[0][1]
    speaker_annos = []
    total_files = sum([len(files) for r, d, files in os.walk(parent_dir)])
    # 2023/4/21: Get the target sampling rate
    with open("./configs/finetune_speaker.json", 'r', encoding='utf-8') as f:
        hps = json.load(f)
    target_sr = hps['data']['sampling_rate']
    processed_files = 0
    for speaker in speaker_names:
        for i, wavfile in enumerate(list(os.walk(parent_dir + speaker))[0][2]):
            # try to load file as audio
            if wavfile.startswith("processed_"):
                continue
            try:
                wav, sr = load_audio(parent_dir + speaker + "/" + wavfile, mono=True)
                if sr != target_sr:
                    from audio_io import resample as _resample
                    wav = _resample(wav, sr, target_sr)
                if wav.shape[1] / target_sr > 20:
                    print(f"{wavfile} is longer than 20s and would be truncated, ignoring\n")
                    continue
                save_path = parent_dir + speaker + "/" + f"processed_{i}.wav"
                save_audio(save_path, wav, target_sr)
                # transcribe text
                lang, text = transcribe_one(model, save_path)
                if lang not in list(lang2token.keys()):
                    print(f"{lang} not supported, ignoring\n")
                    continue
                text = lang2token[lang] + text + lang2token[lang] + "\n"
                speaker_annos.append(save_path + "|" + speaker + "|" + text)

                processed_files += 1
                print(f"Processed: {processed_files}/{total_files}")
            except Exception as exc:  # noqa: BLE001 - keep going through the batch
                print(f"Skipping {wavfile}: {exc!r}")
                continue

    # write into annotation
    if len(speaker_annos) == 0:
        print("Warning: no short audios found, this IS expected if you have only uploaded long audios, videos or video links.")
        print("this IS NOT expected if you have uploaded a zip file of short audios. Please check your file structure or make sure your audio language is supported.")
    with open("short_character_anno.txt", 'w', encoding='utf-8') as f:
        for line in speaker_annos:
            f.write(line)
