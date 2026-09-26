# 本地运行 / Train locally

> 2026 版环境搭建说明见 **[SETUP_2026.md](SETUP_2026.md)**（含 Colab、Windows、Linux 三种方案与完整改动清单）。
> 本文只保留完整的操作流程。

## 0. 环境准备

需要：

* **Python 3.10 – 3.13**（推荐 3.12）
* **ffmpeg**（在 PATH 上）
* 可选：**C/C++ 编译器**（用于编译 `monotonic_align`；没有也能跑，会自动回退到 Numba）
* 可选：**CUDA 12.x** 驱动（有 NVIDIA GPU 时）

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

## 1. 安装 PyTorch

版本要和你的 CUDA 匹配，**先装 torch，再装项目依赖**：

```bash
# Linux + CUDA 12.8
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# Windows / 无 GPU
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Apple Silicon
pip install torch torchaudio
```

> 不要只 `pip install torch`：PyPI 上的 Linux 版会连带装一整套 CUDA 运行时，
> 而你本机驱动未必匹配。用上面的 index-url 明确指定。

## 2. 安装项目依赖

```bash
pip install -r requirements.txt        # 核心：训练 + 推理
pip install -r requirements-data.txt   # 数据处理：Whisper / Demucs / moviepy
```

`requirements-data.txt` 只在需要「从原始音视频自动生成标注」时才用。
已经有标注好的数据的话可以跳过。

## 3. 编译单调对齐扩展（训练必需）

```bash
cd monotonic_align
python setup.py build_ext --inplace
cd ..
```

成功会生成 `monotonic_align/core.*.pyd`（Windows）或 `core.*.so`（Linux/macOS）。

**编译失败也没关系**：代码会自动回退到 Numba 后端，训练照常，只是慢一些。
实际使用的后端会在训练日志里打印：`monotonic_align backend: cython` / `numba` / `python (slow)`。

> Windows 上失败通常是缺 **Visual Studio Build Tools**：安装时勾选
> 「使用 C++ 的桌面开发」工作负载，并确保包含 **Windows SDK**
> （只装编译器不装 SDK 会出现 `cannot open include file: 'io.h'`）。

## 4. 环境自检

```bash
python scripts/verify_install.py
```

退出码为 0 即可开始。该脚本会在真实模型上跑一次完整的前向 + 反向，
几分钟内就能暴露大部分环境问题，建议每次重装环境后都跑一遍。

## 5. 下载预训练底模

可选三档：

```
CJE: 中・日・英 三语
CJ : 中・日 双语
C : 纯中文
```

### Linux / macOS

`CJE`：

```bash
mkdir -p pretrained_models configs
wget https://huggingface.co/spaces/Plachta/VITS-Umamusume-voice-synthesizer/resolve/main/pretrained_models/D_trilingual.pth -O ./pretrained_models/D_0.pth
wget https://huggingface.co/spaces/Plachta/VITS-Umamusume-voice-synthesizer/resolve/main/pretrained_models/G_trilingual.pth -O ./pretrained_models/G_0.pth
wget https://huggingface.co/spaces/Plachta/VITS-Umamusume-voice-synthesizer/resolve/main/configs/uma_trilingual.json -O ./configs/finetune_speaker.json
```

`CJ`：

```bash
wget https://huggingface.co/spaces/sayashi/vits-uma-genshin-honkai/resolve/main/model/D_0-p.pth -O ./pretrained_models/D_0.pth
wget https://huggingface.co/spaces/sayashi/vits-uma-genshin-honkai/resolve/main/model/G_0-p.pth -O ./pretrained_models/G_0.pth
wget https://huggingface.co/spaces/sayashi/vits-uma-genshin-honkai/resolve/main/model/config.json -O ./configs/finetune_speaker.json
```

`C`：

```bash
wget https://huggingface.co/datasets/Plachta/sampled_audio4ft/resolve/main/VITS-Chinese/D_0.pth -O ./pretrained_models/D_0.pth
wget https://huggingface.co/datasets/Plachta/sampled_audio4ft/resolve/main/VITS-Chinese/G_0.pth -O ./pretrained_models/G_0.pth
wget https://huggingface.co/datasets/Plachta/sampled_audio4ft/resolve/main/VITS-Chinese/config.json -O ./configs/finetune_speaker.json
```

### Windows

手动下载上面任一组里的 `G_0.pth`、`D_0.pth`、`config.json`，然后：

* `G`/`D` 模型重命名为 `G_0.pth` / `D_0.pth`，放进 `pretrained_models/`
* 配置文件重命名为 `finetune_speaker.json`，放进 `configs/`

> 切换底模会覆盖上一份，不要同时放多套。

## 6. 准备辅助训练数据（可选）

辅助数据是从预训练大数据集抽样得到的，用于样本少 / 质量差时防止过拟合。
只有在要给 `preprocess_v2.py` 加 `--add_auxiliary_data True` 时才需要：

```bash
wget https://huggingface.co/datasets/Plachta/sampled_audio4ft/resolve/main/sampled_audio4ft_v2.zip
unzip sampled_audio4ft_v2.zip
```

同时建立工作目录：

```bash
mkdir -p video_data raw_audio denoised_audio custom_character_voice segmented_character_voice OUTPUT_MODEL
```

## 7. 放入你的语音数据

命名规则见 [DATA.MD](DATA.MD) / [DATA_EN.MD](DATA_EN.MD)。

**短音频**（单个建议 < 20 秒）

1. 按文档整理成一个 `.zip`
2. 放到 `./custom_character_voice/`
3. 解压：`unzip ./custom_character_voice/custom_character_voice.zip -d ./custom_character_voice/`

**长音频**（单个 ≤ 20 分钟）

按命名规则改好名字，放进 `./raw_audio/`

**视频**（单个 ≤ 20 分钟）

按命名规则改好名字，放进 `./video_data/`

## 8. 处理音频数据

```bash
python scripts/video2audio.py
python scripts/denoise_audio.py
python scripts/long_audio_transcribe.py  --languages "{PRETRAINED_MODEL}" --whisper_size large-v2
python scripts/short_audio_transcribe.py --languages "{PRETRAINED_MODEL}" --whisper_size large-v2
python scripts/resample.py
```

把 `"{PRETRAINED_MODEL}"` 换成 `CJ` / `CJE` / `C` 之一（和上一步选的底模一致）。

* Whisper 跑 `large-v2` 需要 ≥ 12GB 显存；不够就换 `medium` 或 `small`。
* `denoise_audio.py` 需要 `demucs`。没装时会打印警告并跳过人声分离、直接用原始音频，不会中断。
* 没有上传对应类型的素材时，各脚本会友好跳过。

## 9. 生成标注与配置

```bash
# 加辅助数据（适合样本少 / 质量一般；CJ 底模 + 辅助数据通常是中日双语最佳组合）
python preprocess_v2.py --add_auxiliary_data True --languages "{PRETRAINED_MODEL}"

# 不加辅助数据（适合样本多 / 质量高 / 想加快训练）
python preprocess_v2.py --languages "{PRETRAINED_MODEL}"
```

会生成 `final_annotation_train.txt`、`final_annotation_val.txt`
和 `configs/modified_finetune_speaker.json`。

## 10. 开始训练

```bash
python finetune_speaker_v2.py -m ./OUTPUT_MODEL --max_epochs "200" --drop_speaker_embed True
```

`--max_epochs` 填你想要的轮数，经验上 100 以上比较稳。

**续训**（前提：`./OUTPUT_MODEL/` 下有 `G_latest.pth` 与 `D_latest.pth`）：

```bash
python finetune_speaker_v2.py -m ./OUTPUT_MODEL --max_epochs "200" --drop_speaker_embed False --cont True
```

查看训练进度：

```bash
tensorboard --logdir=./OUTPUT_MODEL
# 浏览器打开 localhost:6006
```

> 没有 GPU 也能启动（会降级为单进程 CPU），但速度只适合烟雾测试，不适合真正训练。

## 11. 试听 / 合成

启动网页 UI（TTS 合成 + 音色转换）：

```bash
python scripts/rearrange_speaker.py
cp ./configs/modified_finetune_speaker.json ./finetune_speaker.json
python VC_inference.py --model_dir ./OUTPUT_MODEL/G_latest.pth
```

浏览器打开 `http://127.0.0.1:7860`。

命令行直接合成：

```bash
python cmd_inference.py \
  -m ./OUTPUT_MODEL/G_latest.pth \
  -c ./finetune_speaker.json \
  -t "今天天气不错。" \
  -s "<你的角色名>" \
  -l "简体中文" \
  -o ./output
```

`scripts/rearrange_speaker.py` 会重新编号说话人，并输出可直接分发的
`finetune_speaker.json`、`moegoe_config.json` 和 `G_latest.pth`。

## 12. 清空已处理的数据

### Linux / macOS

```bash
rm -rf ./custom_character_voice/* ./video_data/* ./raw_audio/* ./denoised_audio/* \
       ./segmented_character_voice/* ./separated/* \
       long_character_anno.txt short_character_anno.txt \
       final_annotation_train.txt final_annotation_val.txt
```

### Windows（PowerShell）

```powershell
Remove-Item -Recurse -Force .\custom_character_voice\*, .\video_data\*, .\raw_audio\*,
  .\denoised_audio\*, .\segmented_character_voice\*, .\separated\*
Remove-Item -Force long_character_anno.txt, short_character_anno.txt,
  final_annotation_train.txt, final_annotation_val.txt
```

> 训练产物 `OUTPUT_MODEL/` 不受影响；要清掉请自行删除该目录。
