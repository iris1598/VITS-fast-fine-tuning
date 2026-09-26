import json
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
import whisper  # noqa: E402

from audio_io import load_audio, save_audio  # noqa: E402

parent_dir = "./denoised_audio/"

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
    if not torch.cuda.is_available():
        print("WARNING: no GPU detected. Whisper will run on CPU and will be slow; "
              "consider --whisper_size small or medium.")
    with open("./configs/finetune_speaker.json", 'r', encoding='utf-8') as f:
        hps = json.load(f)
    target_sr = hps['data']['sampling_rate']
    model = whisper.load_model(args.whisper_size)

    if not os.path.isdir(parent_dir):
        print(f"{parent_dir} does not exist — did the previous step run?")
        filelist = []
    else:
        filelist = list(os.walk(parent_dir))[0][2]

    speaker_annos = []
    for file in filelist:
        audio_path = os.path.join(parent_dir, file)
        print(f"Transcribing {audio_path}...\n")

        options = dict(beam_size=5, best_of=5)
        transcribe_options = dict(task="transcribe", **options)

        result = model.transcribe(audio_path, word_timestamps=True, **transcribe_options)
        segments = result["segments"]
        lang = result['language']
        if lang not in lang2token:
            print(f"{lang} not supported, ignoring...\n")
            continue

        character_name = file.rstrip(".wav").split("_")[0]
        code = file.rstrip(".wav").split("_")[1]
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
            print(f"Transcribed segment: {speaker_annos[-1]}")
            save_audio(savepth, wav_seg, target_sr)

    if len(speaker_annos) == 0:
        print("Warning: no long audios & videos found, this IS expected if you have only uploaded short audios")
        print("this IS NOT expected if you have uploaded any long audios, videos or video links. Please check your file structure or make sure your audio/video language is supported.")
    with open("./long_character_anno.txt", 'w', encoding='utf-8') as f:
        for line in speaker_annos:
            f.write(line)
