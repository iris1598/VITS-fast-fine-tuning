import os
import argparse
import json
import sys

sys.setrecursionlimit(500000)  # Fix the error message of RecursionError: maximum recursion depth exceeded while calling a Python object.  You can change the number as you want.

from utils import str2bool


def main():
    parser = argparse.ArgumentParser()
    # NOTE: `type=bool` used to be used here, which made `--add_auxiliary_data
    # False` evaluate to True (any non-empty string is truthy).  str2bool
    # parses the value properly.
    parser.add_argument("--add_auxiliary_data", type=str2bool, nargs="?", const=True,
                        default=False, help="Whether to add extra data as fine-tuning helper")
    parser.add_argument("--languages", default="CJE")
    args = parser.parse_args()

    if args.languages == "CJE":
        langs = ["[ZH]", "[JA]", "[EN]"]
    elif args.languages == "CJ":
        langs = ["[ZH]", "[JA]"]
    elif args.languages == "C":
        langs = ["[ZH]"]
    else:
        raise ValueError(f"Unknown --languages {args.languages!r}; expected one of CJE / CJ / C")

    new_annos = []
    # Source 1: transcribed short audios
    if os.path.exists("short_character_anno.txt"):
        with open("short_character_anno.txt", 'r', encoding='utf-8') as f:
            short_character_anno = f.readlines()
            new_annos += short_character_anno
    # Source 2: transcribed long audio segments
    if os.path.exists("./long_character_anno.txt"):
        with open("./long_character_anno.txt", 'r', encoding='utf-8') as f:
            long_character_anno = f.readlines()
            new_annos += long_character_anno

    # Get all speaker names
    # NOTE: do NOT name this loop variable `text` — it would shadow the `text`
    # module imported below and break `text._clean_text(...)` further down.
    speakers = []
    for line in new_annos:
        path, speaker, txt = line.split("|")
        if speaker not in speakers:
            speakers.append(speaker)
    assert (len(speakers) != 0), "No audio file found. Please check your uploaded file structure."

    # Load the base config once
    with open("./configs/finetune_speaker.json", 'r', encoding='utf-8') as f:
        hps = json.load(f)

    # Make sure every cleaner the config asks for is actually installed.  The
    # text frontends are imported lazily, so without this check a missing
    # backend would only blow up in the middle of the annotation pass.
    import text
    missing = text.check_cleaners(hps['data']['text_cleaners'])
    if missing:
        print("ERROR: the configured text cleaner cannot run:\n  " + "\n  ".join(missing))
        print("Install the missing backend(s) and re-run this step.")
        sys.exit(1)

    if args.add_auxiliary_data:
        with open("./sampled_audio4ft.txt", 'r', encoding='utf-8') as f:
            old_annos = f.readlines()
        # filter old_annos according to supported languages
        filtered_old_annos = []
        for line in old_annos:
            for lang in langs:
                if lang in line:
                    filtered_old_annos.append(line)
        old_annos = filtered_old_annos
        for line in old_annos:
            path, speaker, txt = line.split("|")
            if speaker not in speakers:
                speakers.append(speaker)
        num_old_voices = len(old_annos)
        num_new_voices = len(new_annos)
        # STEP 1: balance number of new & old voices
        cc_duplicate = num_old_voices // num_new_voices
        if cc_duplicate == 0:
            cc_duplicate = 1

        # STEP 2: modify config file
        speaker2id = {}
        for i, speaker in enumerate(speakers):
            speaker2id[speaker] = i
        hps['data']["n_speakers"] = len(speakers)
        hps['speakers'] = speaker2id
        hps['train']['log_interval'] = 10
        hps['train']['eval_interval'] = 100
        hps['train']['batch_size'] = 16
        hps['data']['training_files'] = "final_annotation_train.txt"
        hps['data']['validation_files'] = "final_annotation_val.txt"
        with open("./configs/modified_finetune_speaker.json", 'w', encoding='utf-8') as f:
            json.dump(hps, f, indent=2)

        # STEP 3: clean annotations, replace speaker names with assigned speaker IDs
        cleaned_new_annos = []
        for i, line in enumerate(new_annos):
            path, speaker, txt = line.split("|")
            if len(txt) > 150:
                continue
            cleaned_text = text._clean_text(txt, hps['data']['text_cleaners'])
            cleaned_text += "\n" if not cleaned_text.endswith("\n") else ""
            cleaned_new_annos.append(path + "|" + str(speaker2id[speaker]) + "|" + cleaned_text)
        cleaned_old_annos = []
        for i, line in enumerate(old_annos):
            path, speaker, txt = line.split("|")
            if len(txt) > 150:
                continue
            cleaned_text = text._clean_text(txt, hps['data']['text_cleaners'])
            cleaned_text += "\n" if not cleaned_text.endswith("\n") else ""
            cleaned_old_annos.append(path + "|" + str(speaker2id[speaker]) + "|" + cleaned_text)
        # merge with old annotation
        final_annos = cleaned_old_annos + cc_duplicate * cleaned_new_annos
        with open("./final_annotation_train.txt", 'w', encoding='utf-8') as f:
            for line in final_annos:
                f.write(line)
        with open("./final_annotation_val.txt", 'w', encoding='utf-8') as f:
            for line in cleaned_new_annos:
                f.write(line)
        print("finished")
    else:
        # Do not add extra helper data
        # STEP 1: modify config file
        speaker2id = {}
        for i, speaker in enumerate(speakers):
            speaker2id[speaker] = i
        hps['data']["n_speakers"] = len(speakers)
        hps['speakers'] = speaker2id
        hps['train']['log_interval'] = 10
        hps['train']['eval_interval'] = 100
        hps['train']['batch_size'] = 16
        hps['data']['training_files'] = "final_annotation_train.txt"
        hps['data']['validation_files'] = "final_annotation_val.txt"
        with open("./configs/modified_finetune_speaker.json", 'w', encoding='utf-8') as f:
            json.dump(hps, f, indent=2)

        # STEP 2: clean annotations, replace speaker names with assigned speaker IDs
        import text

        cleaned_new_annos = []
        for i, line in enumerate(new_annos):
            path, speaker, txt = line.split("|")
            if len(txt) > 150:
                continue
            cleaned_text = text._clean_text(txt, hps['data']['text_cleaners']).replace("[ZH]", "")
            cleaned_text += "\n" if not cleaned_text.endswith("\n") else ""
            cleaned_new_annos.append(path + "|" + str(speaker2id[speaker]) + "|" + cleaned_text)

        final_annos = cleaned_new_annos
        with open("./final_annotation_train.txt", 'w', encoding='utf-8') as f:
            for line in final_annos:
                f.write(line)
        with open("./final_annotation_val.txt", 'w', encoding='utf-8') as f:
            for line in cleaned_new_annos:
                f.write(line)
        print("finished")


if __name__ == "__main__":
    main()
