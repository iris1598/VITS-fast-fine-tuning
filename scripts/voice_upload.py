import shutil
import os
import sys
import argparse

try:
    from google.colab import files
except ImportError:  # running outside Colab
    files = None


def _require_colab():
    if files is None:
        print(
            "This script opens an interactive file-upload widget, which only "
            "exists inside Google Colab.\n"
            "When running locally, just copy your files into the target folder:\n"
            "  ./custom_character_voice/  (or ./raw_audio/, ./video_data/)\n"
            "See LOCAL.md for the expected folder layout."
        )
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, required=True, help="type of file to upload")
    args = parser.parse_args()
    file_type = args.type

    _require_colab()

    basepath = os.getcwd()
    uploaded = files.upload()  # 上传文件
    assert (file_type in ['zip', 'audio', 'video'])
    if file_type == "zip":
        upload_path = "./custom_character_voice/"
        os.makedirs(upload_path, exist_ok=True)
        for filename in uploaded.keys():
            #将上传的文件移动到指定的位置上
            shutil.move(os.path.join(basepath, filename), os.path.join(upload_path, "custom_character_voice.zip"))
    elif file_type == "audio":
        upload_path = "./raw_audio/"
        os.makedirs(upload_path, exist_ok=True)
        for filename in uploaded.keys():
            #将上传的文件移动到指定的位置上
            shutil.move(os.path.join(basepath, filename), os.path.join(upload_path, filename))
    elif file_type == "video":
        upload_path = "./video_data/"
        os.makedirs(upload_path, exist_ok=True)
        for filename in uploaded.keys():
            # 将上传的文件移动到指定的位置上
            shutil.move(os.path.join(basepath, filename), os.path.join(upload_path, filename))
