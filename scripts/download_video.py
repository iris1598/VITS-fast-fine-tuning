import os
import random
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
import subprocess

try:
    from google.colab import files
except ImportError:  # running outside Colab
    files = None

basepath = os.getcwd()
if files is None:
    print(
        "This script opens an interactive file-upload widget, which only "
        "exists inside Google Colab.\n"
        "When running locally, write the speaker|video_url pairs into "
        "./speaker_links.txt and run the download part directly."
    )
    sys.exit(1)

uploaded = files.upload()  # 上传文件
for filename in uploaded.keys():
    assert (filename.endswith(".txt")), "speaker-videolink info could only be .txt file!"
    shutil.move(os.path.join(basepath, filename), os.path.join("./speaker_links.txt"))


def generate_infos():
    infos = []
    with open("./speaker_links.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        line = line.replace("\n", "").replace(" ", "")
        if line == "":
            continue
        speaker, link = line.split("|")
        filename = speaker + "_" + str(random.randint(0, 1000000))
        infos.append({"link": link, "filename": filename})
    return infos


def download_video(info):
    link = info["link"]
    filename = info["filename"]
    print(f"Starting download for:\nFilename: {filename}\nLink: {link}")

    try:
        result = subprocess.run(
            ["yt-dlp", "-f", "30280", link, "-o", f"./video_data/{filename}.mp4", "--no-check-certificate"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        print(f"Download completed for {filename}:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to download {link}:\n{e.stderr}")


if __name__ == "__main__":
    os.makedirs("./video_data", exist_ok=True)
    infos = generate_infos()
    if not infos:
        print("speaker_links.txt did not contain any 'speaker|url' lines.")
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        list(executor.map(download_video, infos))
