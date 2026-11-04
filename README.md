# 听声 Ears · 给 AI 装一双"听音乐的耳朵"

把一段音频变成结构化信息：**说了什么、什么调、谁在唱、什么情绪、音高/音色长什么样**。
五个小脚本，纯本地运行，不联网（下载完模型之后）。

> 定位：给 LLM Agent 用的「听觉工具集」。不追求 SOTA，追求**可解释、能跑在普通 CPU 上、结构化可读**。

---

## 能做什么

| 脚本 | 输入 → 输出 | 依赖模型 |
|---|---|---|
| `ears.py` | 音频 → 逐句文字（带时间轴）+ 音高 + 停顿 | whisper-small |
| `music.py` | 音乐 → 调性 / 速度 / 音高 / 音色 / 响度 / **情绪规则判定** | 无（纯 librosa） |
| `emotion.py` | 语音 → 中性 / 开心 / 生气 / 悲伤（占比 + 时间轴） | wav2vec2 SUPERB ER |
| `who.py` | 音频 → **认人**（给参考样本）或**聚类**（看有几个音色） | ECAPA-TDNN |
| `listen.py` | 以上四层的**综合报告** | 按需 |

---

## 快速开始

```bash
# 1) 依赖
pip install -r requirements.txt
# ffmpeg 需要单独装，并保证在 PATH 里

# 2) 下模型（约 900 MB，国内会走 hf-mirror）
python download_models.py

# 3) 跑
python listen.py 你的音频.wav --asr            # 综合报告（含转写）
python music.py  一首歌.wav --json             # 只要音乐特征
python emotion.py 一段说话.wav --json          # 只要情绪
python who.py 一段音频.wav --refs refs/ --json # 认人（refs 里文件名即人名）
```

模型目录默认是脚本旁边的 `models/`，也可以用 `LISTEN_MODELS=/path/to/models` 指定。

---

## 输出长什么样

`music.py`：

```json
{
  "duration": 256.0,
  "key": { "estimate": "G# 小调", "confidence": 0.705, "reliable": true },
  "tempo_bpm": 129.2,
  "tempo_candidates": [64.6, 129.2],
  "pitch": { "median_hz": 207.5, "range_semitones": 23.8, "voiced_ratio": 0.343 },
  "timbre": { "centroid_hz": 2629, "flatness": 0.1396 },
  "loudness": { "dynamic_range_db": 57.1 },
  "mood": { "label": "激烈/热血", "confidence": "high",
            "dims": { "energy": 0.86, "brightness": 0.57 },
            "reasons": ["速度 129 BPM 很快", "小调（偏暗/偏忧）", …] }
}
```

`who.py`（给参考样本时）：

```json
{
  "mode": "identify",
  "refs": ["歌手A", "歌手B"],
  "speakers": [{ "who": "歌手B", "seconds": 257.5 }],
  "segments": [{ "start": 0.0, "end": 2.5, "who": "歌手B", "sim": 0.61 }]
}
```

---

## 几个"实测才知道"的坑（重要）

这些不是理论，是把项目跑歪之后修出来的：

1. **混音直接分析不可靠**：同一首歌，分离后的人声 `voiced_ratio=0.34`，直接喂混音只有 `0.011`，
   音高全是错的。所以先做人声分离（如 UVR / MSST）再分析人声轨。
2. **音高起伏别用标准差**：混音里的八度跳变会把 std 拉到 9 个半音以上（把抒情歌判成"激烈"）。
   本项目最终**只让可靠特征参与判断**（速度、调性置信、音色亮度），音高起伏只报数。
3. **声纹对静音极其敏感**：同一人被切成含静音的窗口，余弦相似度会掉到 **-0.002**（看起来像模型坏了）。
   必须按响度过滤，参考样本也只用最响的几段。
4. **短音频窗口要自适应**：3 秒的语音装不下 4 秒窗口，`range(0, len-win)` 是空的 → 至少保一窗。
5. **Windows 上 speechbrain 会建符号链接**（普通账户 `WinError 1314`）→ 代码里把 `symlink_to` 兜底成复制；
   另外它 `hyperparams.yaml` 里写死了仓库名，会绕过本地文件去重下，`download_models.py` 已自动改掉。
6. **Whisper 中文常出繁体**（"天氣不錯"）→ 用 opencc 统一转简体。
7. **模型下载**：HuggingFace 直连常常不通；`HF_ENDPOINT=https://hf-mirror.com` 是基本操作，
   大文件还会"假死"（进度几分钟不涨）→ 一定要支持断点续传 + 重试。

---

## 隐私提醒

- `who.py` 处理的是**声纹**（生物特征）。请只在获得对方同意的前提下分析他人语音，不要公开他人声音样本。
- `refs/` 里放的是你用来"认人"的参考音频，**不要提交进 Git**（`.gitignore` 已排除）。

---

## 许可与致谢

本项目代码：**MIT**（见 `LICENSE`）。

它站在这些项目肩膀上（各自许可见 `NOTICE.md`）：

| 组件 | 用途 | 许可 |
|---|---|---|
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) + Whisper 模型 | 语音转写 | MIT |
| [speechbrain](https://speechbrain.github.io/) · spkrec-ecapa-voxceleb | 声纹 | Apache-2.0 |
| [SUPERB](https://superbenchmark.org/) · wav2vec2-base-superb-er | 情感 | Apache-2.0 |
| [librosa](https://librosa.org/) | 音频特征 | ISC |
| [opencc](https://github.com/BYVoid/OpenCC) | 简繁转换 | Apache-2.0 |
| [ffmpeg](https://ffmpeg.org/) | 解码 | LGPL/GPL（本项目只调用外部命令，不打包其二进制） |

**注意**：本项目**不分发模型权重**，只提供下载脚本；使用模型时请遵守各自许可。

---

## 自带的验收集

```
demo/demo_speech.wav          系统 TTS 合成的示例语音（不含任何真人声纹）
demo/expected_text_only.json  --text-only 的预期输出
demo/expected_full.json       完整分析（含音乐/情感段）的预期输出
```

装完先跑这两个，能对上就说明环境没问题：

```bash
python listen.py demo/demo_speech.wav --json --text-only
python listen.py demo/demo_speech.wav --json
```

> 转写不会逐字一致（模型版本差异），判据是**语义对得上 + 结构完整**。

---

## 如果你打算把这个仓库丢给 AI 装

根目录有一份 **`AGENT.md`**，专门写给 AI 助手看：分步命令、每步的成功判据、
4 个已知坑的修法、以及"不要做的事"（别提交模型、别公开他人声纹等）。

直接把仓库发给你的 AI，让它读 `AGENT.md` 即可。

---

## 作者

ouerta · MIT License

---

## 实测补充（2026-11-04，接入 QQ 语音后）

把工具接进 QQ 私聊后踩到的四个真实问题，代码里都已处理：

| 问题 | 现象 | 处理 |
|---|---|---|
| **语音是 SILK**（扩展名却是 `.amr`） | ffmpeg 报 `Invalid data` | 让网关 `get_record(out_format=wav)` 转码，别本地硬解 |
| **模型下载 SSL 失败** | `CERTIFICATE_VERIFY_FAILED`（无代理环境） | 强制用本地模型 + `HF_HUB_OFFLINE=1`，不要联网 |
| **长语音缺音频** | 1 分钟语音只有文本占位 `[语音] 1'0"` | 网关不给音频就分析不了，属上游限制 |
| **同步调用超时** | 单条分析 18–24 秒，撞工具调用上限 | 改成异步任务：立刻返回 jobId，再查结果（声纹默认关闭再省 5–8 秒） |

另外：**短语音（3–10 秒）转写很准**（实测「一二三四五六」全对）；
`emotion.py` 对偏轻的语音有兜底（全被响度门限滤掉时取最响的几窗，并标 `lowVolume`）。
