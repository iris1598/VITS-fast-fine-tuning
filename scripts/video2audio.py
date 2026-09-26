import os
from concurrent.futures import ThreadPoolExecutor

try:  # moviepy >= 2.0
    from moviepy import AudioFileClip
except ImportError:  # moviepy < 2.0
    from moviepy.editor import AudioFileClip

video_dir = "./video_data/"
audio_dir = "./raw_audio/"

VIDEO_EXT = {".mp4", ".mkv", ".webm", ".mov", ".flv", ".avi"}


def generate_infos():
    if not os.path.isdir(video_dir):
        return [], []
    filelist = sorted(os.listdir(video_dir))
    videos = [f for f in filelist
              if os.path.isfile(os.path.join(video_dir, f))
              and os.path.splitext(f)[1].lower() in VIDEO_EXT]
    non_mp4 = [f for f in videos if not f.lower().endswith(".mp4")]
    return videos, non_mp4


def clip_file(file):
    os.makedirs(audio_dir, exist_ok=True)
    # NOTE: os.path.splitext, not str.rstrip("mp4") — rstrip treats its argument
    # as a character set and would eat trailing m/p/4 characters of the stem.
    stem = os.path.splitext(file)[0]
    out_path = os.path.join(audio_dir, stem + ".wav")
    my_audio_clip = AudioFileClip(os.path.join(video_dir, file))
    try:
        my_audio_clip.write_audiofile(out_path)
    finally:
        my_audio_clip.close()
    print(f"  {file}  ->  raw_audio/{stem}.wav")


if __name__ == "__main__":
    infos, non_mp4 = generate_infos()
    if not infos:
        print(f"No video files found under {video_dir} — nothing to extract "
              "(this is expected if you only uploaded audio).")
    else:
        print(f"从 {len(infos)} 个视频中抽取音频...")
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            list(executor.map(clip_file, infos))
        if non_mp4:
            print(f"\n⚠️ 以下视频不是 .mp4，可能无法被后续步骤处理：{non_mp4}")
            print("   DATA.MD 要求使用 .mp4 格式。")
        print(f"\n✅ 音频已写入 {audio_dir}（文件名沿用视频名，"
              f"如 Taffy_332452.mp4 -> Taffy_332452.wav）")
