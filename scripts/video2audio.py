import os
from concurrent.futures import ThreadPoolExecutor

try:  # moviepy >= 2.0
    from moviepy import AudioFileClip
except ImportError:  # moviepy < 2.0
    from moviepy.editor import AudioFileClip

video_dir = "./video_data/"
audio_dir = "./raw_audio/"


def generate_infos():
    if not os.path.isdir(video_dir):
        return []
    filelist = list(os.walk(video_dir))[0][2]
    return [f for f in filelist if f.endswith(".mp4")]


def clip_file(file):
    os.makedirs(audio_dir, exist_ok=True)
    my_audio_clip = AudioFileClip(video_dir + file)
    try:
        my_audio_clip.write_audiofile(audio_dir + file.rstrip("mp4") + "wav")
    finally:
        my_audio_clip.close()


if __name__ == "__main__":
    infos = generate_infos()
    if not infos:
        print(f"No .mp4 files found under {video_dir} — nothing to extract "
              "(this is expected if you only uploaded audio).")
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        list(executor.map(clip_file, infos))
