import torch
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compat import torch_load  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="./OUTPUT_MODEL/G_latest.pth")
    parser.add_argument("--config_dir", type=str, default="./configs/modified_finetune_speaker.json")
    args = parser.parse_args()

    model_sd = torch_load(args.model_dir, map_location='cpu')
    with open(args.config_dir, 'r', encoding='utf-8') as f:
        hps = json.load(f)

    valid_speakers = list(hps['speakers'].keys())
    if hps['data']['n_speakers'] > len(valid_speakers):
        new_emb_g = torch.zeros([len(valid_speakers), 256])
        old_emb_g = model_sd['model']['emb_g.weight']
        for i, speaker in enumerate(valid_speakers):
            new_emb_g[i, :] = old_emb_g[hps['speakers'][speaker], :]
            hps['speakers'][speaker] = i
        hps['data']['n_speakers'] = len(valid_speakers)
        model_sd['model']['emb_g.weight'] = new_emb_g
        with open("./finetune_speaker.json", 'w', encoding='utf-8') as f:
            json.dump(hps, f, indent=2)
        torch.save(model_sd, "./G_latest.pth")
    else:
        with open("./finetune_speaker.json", 'w', encoding='utf-8') as f:
            json.dump(hps, f, indent=2)
        torch.save(model_sd, "./G_latest.pth")
    # save another config file copy in MoeGoe format
    hps['speakers'] = valid_speakers
    with open("./moegoe_config.json", 'w', encoding='utf-8') as f:
        json.dump(hps, f, indent=2)
    print(f"Wrote finetune_speaker.json / moegoe_config.json / G_latest.pth "
          f"({len(valid_speakers)} speakers)")
