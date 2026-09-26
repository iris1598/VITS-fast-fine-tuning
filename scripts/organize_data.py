"""Normalise the uploaded data so the rest of the pipeline can find it.

Two layout problems break the pipeline in practice:

1. **Short audio.** `DATA.MD` wants::

       custom_character_voice/<CharacterName>/*.wav

   but almost everyone zips up a *flat* folder of clips, or a zip that contains
   one wrapper folder.  `short_audio_transcribe.py` only scans per-character
   sub-directories, so a flat layout silently produces zero samples and training
   later fails with "No audio file found".

2. **Long audio / video.** `DATA.MD` wants ``{CharacterName}_{number}.wav`` and
   the transcription step splits the file name on ``_``.  A file simply named
   ``myvoice.wav`` used to raise ``IndexError``.

This script fixes both in place and prints a report of the resulting layout.
It is idempotent — running it twice is harmless.

Usage::

    python scripts/organize_data.py --character myvoice
    python scripts/organize_data.py --character myvoice --dry-run
"""

import argparse
import os
import re
import shutil
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".opus"}
VIDEO_EXT = {".mp4", ".mkv", ".webm", ".mov", ".flv", ".avi"}

SHORT_DIR = "./custom_character_voice"
RAW_DIR = "./raw_audio"
VIDEO_DIR = "./video_data"

# {CharacterName}_{number}, e.g. Taffy_332452
_NUMBERED = re.compile(r"^(?P<name>.+)_(?P<code>\d+)$")


def split_character(stem):
    """``'Taffy_332452'`` -> ``('Taffy', '332452')``.

    A stem without a trailing ``_<digits>`` is treated as the character name
    with code ``0`` — that is what ``long_audio_transcribe.py`` needs, and
    :func:`normalize_names` makes it explicit on disk anyway.
    """
    m = _NUMBERED.match(stem)
    if m:
        return m.group("name"), m.group("code")
    return stem, "0"


def _visible(path):
    return sorted(e for e in os.listdir(path)
                  if not e.startswith(".") and e != "__MACOSX")


def _is_audio(name):
    return os.path.splitext(name)[1].lower() in AUDIO_EXT


def _is_video(name):
    return os.path.splitext(name)[1].lower() in VIDEO_EXT


def _audio_in(path):
    return [f for f in _visible(path)
            if os.path.isfile(os.path.join(path, f)) and _is_audio(f)]


def _move_unique(src, dst):
    """Move ``src`` to ``dst``, adding a numeric suffix instead of overwriting."""
    if os.path.abspath(src) == os.path.abspath(dst):
        return dst
    root, ext = os.path.splitext(dst)
    candidate, i = dst, 1
    while os.path.exists(candidate):
        candidate = f"{root}_{i}{ext}"
        i += 1
    shutil.move(src, candidate)
    return candidate


# ---------------------------------------------------------------------------
def extract_zips(root, dry_run=False):
    """Unpack any ``.zip`` sitting inside ``root`` (the upload flow leaves one).

    The archive is then moved into ``<root>/_uploads/`` so that re-running this
    script does not extract it a second time — otherwise the clips would
    reappear at the top level and get copied into the character folder again,
    leaving ``_1``/``_2`` duplicates behind.
    """
    done = []
    if not os.path.isdir(root):
        return done
    archive_dir = os.path.join(root, "_uploads")
    for entry in _visible(root):
        if not entry.lower().endswith(".zip"):
            continue
        path = os.path.join(root, entry)
        if dry_run:
            done.append(entry)
            continue
        try:
            with zipfile.ZipFile(path) as z:
                z.extractall(root)
            os.makedirs(archive_dir, exist_ok=True)
            _move_unique(path, os.path.join(archive_dir, entry))
            done.append(entry)
        except zipfile.BadZipFile:
            print(f"  ! {entry} 不是有效的 zip，跳过")
    return done


def clean_junk(root):
    """Remove __MACOSX folders and macOS resource-fork files.

    Note: the previous version filtered ``__MACOSX`` out of ``dirnames`` and
    only *then* tried to delete it, so it was never removed. Delete first, then
    drop it from the walk list.
    """
    removed = 0
    if not os.path.isdir(root):
        return removed
    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        for d in list(dirnames):
            if d == "__MACOSX":
                shutil.rmtree(os.path.join(dirpath, d), ignore_errors=True)
                dirnames.remove(d)
                removed += 1
            elif d.startswith("."):
                # never descend into (or delete) hidden dirs such as .git
                dirnames.remove(d)
        for f in filenames:
            if f.startswith("._") or f in (".DS_Store", "Thumbs.db"):
                try:
                    os.remove(os.path.join(dirpath, f))
                    removed += 1
                except OSError:
                    pass
    return removed


def collapse_wrappers(root):
    """``custom_character_voice/<wrapper>/<Character>/*.wav`` -> ``<Character>/*.wav``.

    Only levels that contain no audio of their own are collapsed, so a folder
    that *is* a character folder is never renamed.
    """
    collapsed = []
    if not os.path.isdir(root):
        return collapsed
    for _ in range(5):
        entries = _visible(root)
        if _audio_in(root):
            break
        subdirs = [e for e in entries if os.path.isdir(os.path.join(root, e))]
        if len(subdirs) != 1:
            break
        inner = os.path.join(root, subdirs[0])
        inner_subdirs = [e for e in _visible(inner)
                         if os.path.isdir(os.path.join(inner, e))]
        if not inner_subdirs:
            # the single child holds clips directly -> it already IS a character dir
            break
        for e in _visible(inner):
            _move_unique(os.path.join(inner, e), os.path.join(root, e))
        try:
            os.rmdir(inner)
        except OSError:
            break
        collapsed.append(subdirs[0])
    return collapsed


def place_flat_audio(root, character, dry_run=False):
    """Move clips that sit directly in ``root`` into ``root/<character>/``."""
    flat = _audio_in(root)
    if not flat:
        return 0
    if dry_run:
        return len(flat)
    dest = os.path.join(root, character)
    os.makedirs(dest, exist_ok=True)
    for f in flat:
        _move_unique(os.path.join(root, f), os.path.join(dest, f))
    return len(flat)


def character_dirs(root):
    """``[(name, audio_count), ...]`` for every directory that holds audio."""
    out = []
    if not os.path.isdir(root):
        return out
    root_abs = os.path.abspath(root)
    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        dirnames[:] = [d for d in dirnames if d != "__MACOSX" and not d.startswith(".")]
        if os.path.abspath(dirpath) == root_abs:
            continue
        audio = [f for f in filenames if _is_audio(f)]
        if audio:
            out.append((os.path.relpath(dirpath, root), len(audio)))
    return sorted(out)


def normalize_names(root, exts, dry_run=False):
    """Rename ``<stem>.<ext>`` to ``<stem>_0.<ext>`` when there is no ``_<digits>``."""
    renamed = []
    if not os.path.isdir(root):
        return renamed
    for f in _visible(root):
        path = os.path.join(root, f)
        if not os.path.isfile(path):
            continue
        stem, ext = os.path.splitext(f)
        if ext.lower() not in exts:
            continue
        if _NUMBERED.match(stem):
            continue
        new_name = f"{stem}_0{ext}"
        renamed.append((f, new_name))
        if not dry_run:
            _move_unique(path, os.path.join(root, new_name))
    return renamed


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--character", default="myvoice",
                        help="character name used for short-audio clips that are not "
                             "already inside a per-character folder (default: myvoice)")
    parser.add_argument("--only", choices=["all", "short", "long", "video"], default="all",
                        help="normalise only one kind of input (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would change without touching anything")
    args = parser.parse_args()

    dry = args.dry_run
    prefix = "[dry-run] " if dry else ""
    only = args.only

    print("=" * 70)
    print(f"{prefix}规范化数据目录 / normalizing data layout")
    print("=" * 70)

    n_flat = 0
    hoisted = []

    # --- short audio ------------------------------------------------------
    if only in ("all", "short"):
        print(f"\n[短音频] {SHORT_DIR}")
        if not os.path.isdir(SHORT_DIR):
            print("  目录不存在，跳过")
        else:
            for z in extract_zips(SHORT_DIR, dry):
                print(f"  {prefix}解压 {z}")

            junk = clean_junk(SHORT_DIR)
            if junk:
                print(f"  清理 {junk} 个系统垃圾文件")

            hoisted = collapse_wrappers(SHORT_DIR)
            for h in hoisted:
                print(f"  {prefix}去掉多余的包装目录 {h}/")

            n_flat = place_flat_audio(SHORT_DIR, args.character, dry)
            if n_flat:
                print(f"  {prefix}把 {n_flat} 个平铺音频移动进 {args.character}/")

            dirs = character_dirs(SHORT_DIR)
            if dirs:
                print("  角色目录：")
                for name, cnt in dirs:
                    print(f"    {name}/   {cnt} 个音频")
                if len(dirs) == 1 and dirs[0][1] < 10:
                    print(f"  ⚠️ 只有 {dirs[0][1]} 条短音频，"
                          f"DATA.MD 建议每个角色至少 10 条（最好 20+）")

                # A single folder holding clips directly is structurally identical
                # whether it is a real character folder or just the wrapper created
                # by zipping a folder. We must not guess — renaming a real character
                # folder would be worse than leaving it. So: state what will happen.
                if len(dirs) == 1 and n_flat == 0 and not hoisted:
                    name = dirs[0][0]
                    print(f"  提示：脚本会把 `{name}` 直接当作角色名。")
                    print(f"       如果它其实是压缩包的包装目录（例如 samples/、audio/），"
                          f"请改名后重跑：")
                    print(f"         mv ./custom_character_voice/{name} "
                          f"./custom_character_voice/<你的角色名>")
            else:
                print("  ⚠️ 没有找到任何音频。请确认已上传，或 zip 结构是否正确。")
                print("     如果音频是平铺放进来的，检查 zip 是否解压成功。")

    # --- long audio -------------------------------------------------------
    if only in ("all", "long"):
        print(f"\n[长音频] {RAW_DIR}")
        if not os.path.isdir(RAW_DIR):
            print("  目录不存在，跳过")
        else:
            renamed = normalize_names(RAW_DIR, {".wav"}, dry)
            for a, b in renamed:
                print(f"  {prefix}{a}  ->  {b}")
            if not renamed:
                print("  文件名已符合规范")
            wavs = [f for f in _visible(RAW_DIR) if _is_audio(f)]
            if wavs:
                print(f"  共 {len(wavs)} 个音频：{', '.join(wavs[:4])}"
                      + (" ..." if len(wavs) > 4 else ""))
            else:
                print("  （没有长音频，属正常）")

    # --- video ------------------------------------------------------------
    if only in ("all", "video"):
        print(f"\n[视频] {VIDEO_DIR}")
        if not os.path.isdir(VIDEO_DIR):
            print("  目录不存在，跳过")
        else:
            renamed = normalize_names(VIDEO_DIR, VIDEO_EXT, dry)
            for a, b in renamed:
                print(f"  {prefix}{a}  ->  {b}")
            if not renamed:
                print("  文件名已符合规范")
            vids = [f for f in _visible(VIDEO_DIR) if _is_video(f)]
            non_mp4 = [f for f in vids if not f.lower().endswith(".mp4")]
            if vids:
                print(f"  共 {len(vids)} 个视频")
            else:
                print("  （没有视频，属正常）")
            if non_mp4:
                print(f"  ⚠️ 以下文件不是 .mp4，video2audio.py 只会处理 .mp4：{non_mp4}")

    # --- verdict ----------------------------------------------------------
    print("\n" + "=" * 70)
    short_ok = bool(character_dirs(SHORT_DIR))
    raw_ok = os.path.isdir(RAW_DIR) and bool(
        [f for f in _visible(RAW_DIR) if _is_audio(f)])
    vid_ok = os.path.isdir(VIDEO_DIR) and bool(
        [f for f in _visible(VIDEO_DIR) if _is_video(f)])

    if short_ok or raw_ok or vid_ok:
        parts = []
        if short_ok:
            parts.append(f"短音频 {sum(c for _, c in character_dirs(SHORT_DIR))} 条")
        if raw_ok:
            parts.append("长音频")
        if vid_ok:
            parts.append("视频")
        print("✅ 数据已就绪：" + "、".join(parts))
        if not raw_ok and not vid_ok:
            print("   只检测到短音频 —— STEP 3 里 long_audio/denoise 步骤会正常跳过。")
        return 0

    if only == "short":
        print("❌ 没有找到短音频。请回到 STEP 2.1 上传，再运行本单元格。")
    else:
        print("❌ 三个目录里都没有找到数据。请回到 STEP 2 上传素材。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
