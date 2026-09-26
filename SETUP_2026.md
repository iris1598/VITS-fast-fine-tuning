# 2026 环境搭建与迁移说明

本项目原始代码写于 2023 年，目标环境是 **Python 3.8 / PyTorch 2.1 / NumPy 1.22 / CUDA 11.7**。
现在的运行环境（Google Colab 2026 免费版）是 **Python 3.12 / NumPy 2.x / PyTorch 2.11+ / CUDA 12.8**，
旧代码和旧 notebook 已经无法直接运行。本文说明改了什么、以及现在怎么搭环境。

---

## 快速开始

### 方案 A：Google Colab（推荐）

打开仓库根目录的 `VITS-fast-finetuning.ipynb`，按 STEP 0.1 → 5 顺序运行。
**STEP 0.2 是唯一需要按需修改的单元格**（指定代码来源和 pip 镜像）。

### 方案 B：本地（Windows / Linux）

```bash
# 1) Python 3.12（3.10 - 3.13 均可）
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate

# 2) 先装 PyTorch —— 版本要匹配你的 CUDA
#    Linux + CUDA 12.8
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
#    Windows / 无 GPU
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# 3) 再装项目依赖（核心 + 数据处理分开）
pip install -r requirements.txt
pip install -r requirements-data.txt

# 4) 编译单调对齐扩展（训练必需）
cd monotonic_align && python setup.py build_ext --inplace && cd ..

# 5) 自检
python scripts/verify_install.py
```

还需要 **ffmpeg** 在 PATH 上（处理视频/音频解码）：`apt install ffmpeg` / `brew install ffmpeg` / Windows 从 ffmpeg.org 下载。

> `requirements-data.txt` 里的 Whisper / Demucs 只有「STEP 3 处理原始音视频」才需要。
> 如果你已经有标注好的数据，可以跳过它。

---

## 自检脚本

```bash
python scripts/verify_install.py                 # 用默认的三语配置
python scripts/verify_install.py --config ./configs/modified_finetune_speaker.json
```

它会依次检查并在真实模型上跑一次完整的前向+反向：

| 检查项 | 说明 |
|---|---|
| versions | Python / PyTorch / NumPy / CUDA |
| module imports | 所有模块导入 + `scripts/` 语法 |
| monotonic_align | 后端类型 + 对齐结果的单调性 |
| lazy optional backends | 缺可选语言包时 `import text` 仍可用 |
| text language backends | 中/英/韩/日 各自能否产出音素 |
| config text frontend | 配置的 cleaner 与 symbol 表是否自洽 |
| audio I/O | soundfile 读写与重采样 |
| data pipeline + training step | 数据集 → 批处理 → 生成器/判别器前向、反向、优化器更新 |
| inference | `SynthesizerTrn.infer()` 产出音频 |

退出码 0 = 可以开始训练。`VERIFY_TRACEBACK=1` 可打印完整堆栈。

---

## 到底改了哪些地方

### 1. 依赖（`requirements.txt` 全面重写）

| 旧 | 问题 | 现在 |
|---|---|---|
| `numpy==1.22` | 不支持 Python 3.11+，且无法与 NumPy 2 生态共存 | `numpy>=1.26,<3` |
| `Cython==0.29.21` | 与 Python 3.12 / NumPy 2 不兼容 | `Cython>=3.0` |
| `matplotlib==3.3.1`、`scikit-learn==1.0.2` | 无 3.12 轮子 | 交给 `matplotlib>=3.8`，移除未使用的 sklearn 钉子 |
| `torch==2.1.2` 等 | 会覆盖 Colab 预装的 CUDA 版 torch | **不在 requirements 里钉 torch**，单独安装；notebook 里还用约束文件锁住 |
| `pyopenjtalk-prebuilt` | 无 3.12+ 轮子，需 CMake + MSVC 现场编译 | `pyopenjtalk-plus`（官方预编译轮子，导入名仍是 `pyopenjtalk`） |
| `librosa==0.9.2` | 与 NumPy 2 不兼容 | `librosa>=0.10.2` |
| `pydantic==1.10.4` | 与 gradio 4/5 要求 pydantic 2 冲突 | 移除该钉子 |
| `imageio==2.4.1`（notebook 里） | 与 moviepy 2.x 冲突 | 移除 |
| `demucs` / `whisper` / `gradio` 混在一起 | 任一装坏都会拖垮训练环境 | 拆成 `requirements-data.txt` |
| `opencc`、`indic_transliteration`、`num_thai` | 未安装就导致 `import text` 崩溃 | 改为惰性导入（见第 5 点） |

### 2. `monotonic_align`（训练能否启动的关键）

* `from distutils.core import setup` → **Python 3.12 已删除 `distutils`**，改用 `setuptools`。
* `Extension("monotonic_align.core", ...)` 的旧布局会把产物塞进嵌套目录（所以旧文档要求 `mkdir monotonic_align`）。
  现在改成 `Extension("core", ...)`，产物直接落在 `monotonic_align/` 下。
* 新增 `pyproject.toml` 声明构建依赖。
* **新增两级回退**：Cython 扩展 → Numba JIT → 纯 Python。
  没有 C 编译器（典型场景：Windows 没装 MSVC）时，不会再直接报错，而是降级运行并给出提示。
  当前后端会在训练日志里打印：`monotonic_align backend: numba`。

### 3. PyTorch / NumPy / 音频类 API

| 位置 | 旧写法 | 新写法 |
|---|---|---|
| `finetune_speaker_v2.py` | `from torch.cuda.amp import autocast, GradScaler` | `compat.autocast()` / `compat.make_grad_scaler()`，自动适配 `torch.amp` |
| `utils.py`、`scripts/rearrange_speaker.py` | `torch.load(...)` | `compat.torch_load()`：先试 `weights_only=True`（2.6 起的新默认），失败再回退并告警 |
| `mel_processing.py` | `torch.stft(..., return_complex=False)` | `return_complex=True` + `torch.view_as_real()`，数学等价 |
| `mel_processing.py` | `librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax)` 位置参数 | **librosa 0.10 起全部为关键字参数**，改为 `sr=, n_fft=, n_mels=, fmin=, fmax=` |
| `utils.py` | `np.fromstring(fig.canvas.tostring_rgb(), ...)` | `tostring_rgb` 在 matplotlib 3.10 被删除；改用 `fig.canvas.buffer_rgba()` |
| `modules.py`、`models.py` | 直接 `from torch.nn.utils import weight_norm` | 走 `compat`，**仍使用旧实现**（预训练权重的 `weight_g`/`weight_v` 键名依赖它，换成 parametrizations 会加载失败） |
| `data_utils.py`、`scripts/*` | `torchaudio.load(..., normalize=True)` | torchaudio 2.9 起 I/O 改由 TorchCodec 承担、`normalize` 被忽略且缺 TorchCodec 会直接报错；统一改为 **soundfile 后端**（`audio_io.py`），torchaudio / librosa 作为回退 |
| `VC_inference.py` | `librosa.to_mono` / `librosa.resample` | `audio_io.resample()`（底层 torchaudio transforms，返回 torch 张量） |

新增两个模块：

* **`compat.py`** — 版本兼容层（checkpoint 加载、AMP、weight_norm、噪声告警抑制）
* **`audio_io.py`** — 稳定的音频读写与重采样（`load_audio` / `save_audio` / `resample`）

### 4. 训练脚本（`finetune_speaker_v2.py`）

* **支持无 GPU 运行**：原来是 `assert torch.cuda.is_available()` 硬失败。现在会打印警告并降级到
  单进程 CPU 训练（`gloo` 后端），便于烟雾测试和排障；有 GPU 时行为不变。
* DDP 在 CPU 上不再传 `device_ids`（否则报错）；`MASTER_PORT` 自动选空闲端口，避免并行任务撞端口。
* `autocast` / `GradScaler` 的启用条件与设备绑定，无 GPU 时自动关闭 fp16。
* epoch 调度修正：原来靠 `epoch > max_epochs` 时 `exit()` 强杀进程（会让 DDP 无法正常收尾），
  改为按 `min(train.epochs, --max_epochs)` 正常循环。
* 验证集为空时不再崩溃。

### 5. 文本前端（`text/`）

**这是最影响可用性的一处。** 原来 `text/__init__.py` → `cleaners.py` 在导入时就把
日语 / 韩语 / 泰语 / 梵语 / 英语 全部加载，于是缺 `indic_transliteration` 或 `num_thai`
会导致 `import text` 直接失败 —— **即使你训练的是纯中文模型**。

现在：

* 每个 cleaner 在自己的函数体内惰性导入对应语言模块。
* `text/japanese.py`、`text/thai.py`、`text/sanskrit.py` 在缺依赖时抛出带安装命令的
  `MissingBackend`，例如：`pip install pyopenjtalk-plus`。
* `preprocess_v2.py` 在开始清洗前先校验配置要求的 cleaner 是否可用，提前给出明确提示。
* 顺带修掉 `text/__init__.py` 里每加载一条样本就 `print()` 两次的日志噪声，
  以及 `japanese.py` 中读 `labels[n+1]` 的越界隐患。

### 6. 数据处理脚本（`scripts/`）

* **moviepy 2.x**：`from moviepy.editor import ...` 已废弃，改为 `from moviepy import ...`（保留旧版回退）。
* **torchaudio → soundfile**（同上），并修正 `denoise_audio.py` 里遍历了全部文件而非只处理 `.wav` 的问题。
* `denoise_audio.py`：Demucs 不可用时打印警告并**跳过人声分离**继续，而不是让整条流水线失败。
* `short_audio_transcribe.py`：原来 `except: continue` 会静默吞掉所有错误，现在会打印原因；
  超过 20 秒的样本按文档语义真正跳过（原来只打印「ignoring」但照样处理）。
* `resample.py` / `denoise_audio.py` / `*_transcribe.py`：源目录不存在时给出友好提示而非崩溃。
* `voice_upload.py` / `download_video.py` / `download_model.py`（依赖 `google.colab`）：
  在 Colab 之外运行时给出明确说明，而不是 `ModuleNotFoundError`。
* `rearrange_speaker.py`：`torch.load` 走兼容层；输出成功信息。

### 7. 推理界面（`VC_inference.py`）

* **Gradio 4/5 组件 API**：`gr.Audio(source="microphone")` → `gr.Audio(sources=["microphone"], type="numpy")`；
  `gr.TextArea` → `gr.Textbox(lines=3)`（前者在新版已移除）。
* **音频 dtype 变更**：Gradio 3 的 `type="numpy"` 返回 int16，新版返回 **float32 且已归一化到 [-1, 1]**。
  原代码的 `audio / np.iinfo(audio.dtype).max` 在 float32 上会直接抛异常，已改为按 dtype 判断。
* `--share` 用 `str2bool` 解析，不再依赖字符串真值；非 Colab 环境不再强行 `webbrowser.open`。

### 8. `preprocess_v2.py`

* `--add_auxiliary_data` 原为 `type=bool`，**任何非空字符串都是 True**，所以 `--add_auxiliary_data False`
  实际会开启辅助数据。改为 `str2bool` + `nargs="?"`，默认 False。
* 增加 cleaner 可用性前置校验；未知语言参数给出明确报错。

### 9. 数据管线（`data_utils.py`）

* `get_audio` 换用 soundfile 后端。
* `get_text` 修正了 `text_to_sequence` 的参数个数错误（旧代码少传一个参数，只在
  `cleaned_text=False` 时才会触发）。
* `get_sid` 在说话人 id 非整数时给出可操作的报错，而不是 `invalid literal for int()`。

---

## 常见问题

**Q: 日志出现 `monotonic_align backend: python (slow)` 怎么办？**
说明 Cython 扩展和 Numba 都不可用，训练会极慢。优先编译扩展：
`cd monotonic_align && python setup.py build_ext --inplace`。
Windows 上编译失败通常是缺 Visual Studio Build Tools（需要「使用 C++ 的桌面开发」工作负载 + Windows SDK）。

**Q: 会不会有真正的 CUDA 加速？**
会。本仓库只负责把代码改到能跑；有 GPU 时 `fp16_run`、NCCL、DDP 都按原逻辑生效。

**Q: 我能继续用 2023 年的旧 notebook 吗？**
不能。它钉死的依赖在 Python 3.12 上装不上，`pip install imageio==2.4.1` 也会和 moviepy 冲突。
请使用仓库里的 `VITS-fast-finetuning.ipynb`。

**Q: 训练结果和 2023 年会有差异吗？**
数值路径保持一致：`torch.stft` 的等价改写、soundfile 的浮点归一化方式都与原实现等价，
`weight_norm` 仍用旧实现以保证预训练权重可加载。mel 滤波器仍由 librosa 计算，参数逐项对齐。

**Q: 想确认预训练权重能正常加载？**
在 STEP 1.5 之后跑一次 STEP 1.3 的自检，或直接训练：日志里会打印
`Train with pretrained model...` 且不出现 `is not in the checkpoint` 的刷屏。
