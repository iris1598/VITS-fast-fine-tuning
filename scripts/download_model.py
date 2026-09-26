import os
import sys

try:
    from google.colab import files
except ImportError:  # running outside Colab
    files = None

if files is None:
    print(
        "This script triggers a browser download, which only makes sense inside "
        "Google Colab.\n"
        "The files are already on disk next to this project:\n"
        "  ./G_latest.pth\n"
        "  ./finetune_speaker.json\n"
        "  ./moegoe_config.json"
    )
    sys.exit(1)

for name in ("./G_latest.pth", "./finetune_speaker.json", "./moegoe_config.json"):
    if not os.path.exists(name):
        print(f"WARNING: {name} not found, skipping.")
        continue
    files.download(name)
