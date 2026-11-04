# AGENT.md — 给「拿到这个仓库的 AI」的安装与验收说明

> 这份文件是**写给 AI 助手看的**（不是写给人看的）。如果你是被用户丢过来的 AI：
> 请按顺序执行，每一步都有"怎么验证成功"，不要跳步，不要凭猜。
> 用户可能只说了"把这个装上"，那你也要先做第 0 步。

---

## 第 0 步：先读一遍环境，别急着装

```bash
python --version      # 需要 3.10+
ffmpeg -version       # 需要能跑；没有就装（见下）
```

判断平台：Windows / macOS / Linux 都支持，但**有两处 Windows 特有行为**（见第 4 步的坑）。

如果 `python` 不在 PATH，先在系统里找到它（Windows 上常见 `C:\Python3xx\python.exe` 或某个 venv），
**后续所有命令都用那个解释器**，不要混用。

**不要**在没有用户确认的情况下：
- 修改系统环境变量
- 全局 `pip install` 到系统 Python（优先建 venv：`python -m venv .venv`）
- 下载超过 1 GB 的东西（本项目的模型约 900 MB，先告诉用户再下）

---

## 第 1 步：装依赖

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

**ffmpeg**（本项目只调用它，不打包）：

| 平台 | 命令 |
|---|---|
| Windows | `winget install Gyan.FFmpeg` 或 `choco install ffmpeg` |
| macOS | `brew install ffmpeg` |
| Linux | `sudo apt install ffmpeg` |

验证：`ffmpeg -version` 能打印版本号即成功。

---

## 第 2 步：下模型（约 900 MB）

```bash
python download_models.py
# 或者只下某一个：
python download_models.py --only whisper
python download_models.py --only ecapa
python download_models.py --only emotion
```

会落到 `./models/`：

```
models/whisper-small      语音转文字
models/ecapa              声纹（认人）
models/emotion_superb     情感 4 类
```

**国内网络**：脚本默认已设 `HF_ENDPOINT=https://hf-mirror.com`。
直连 HuggingFace 在国内通常不通；若用户在国外想用官方源，可不设这个环境变量。

验证：

```bash
ls models/            # Windows: dir models
# 三个子目录都在，且 ecapa 里有 embedding_model.ckpt（约 80 MB）
```

---

## 第 3 步：跑 demo 验收（不要跳过）

仓库自带 `demo/demo_speech.wav`（系统 TTS 合成，不含任何真人声纹），以及两份**预期输出**。

```bash
python listen.py demo/demo_speech.wav --json --text-only
```

**成功标准**（对照 `demo/expected_text_only.json`）：

- 输出是合法 JSON
- `speech.text` 里大致包含「这是一段合成语音」「转成文字」这些词
- `summary` 以「语音转写：」开头

再跑完整版：

```bash
python listen.py demo/demo_speech.wav --json
```

**成功标准**（对照 `demo/expected_full.json`）：

- 有 `music` / `emotion` 两段
- `emotion.dominant` 是 中性/开心/生气/悲伤 之一

> 转写**不会逐字一致**（不同模型/版本有差异），不要因为差几个字就判定失败。
> 判据是"语义对得上 + 结构完整"。

---

## 第 4 步：四个已知坑（遇到直接照下面的修）

### 坑 1：`WinError 1314` / 符号链接权限（Windows）

```
OSError: [WinError 1314] 客户端没有所需的特权: ... symlink_to ...
```

**原因**：speechbrain 在 Windows 上默认用符号链接摆模型文件，普通账户没权限。
**修法**：`who.py` 里已经把 `pathlib.Path.symlink_to` 兜底成复制；如果仍报错，
说明还有 `models/ecapa/.cache` 目录，删掉它再跑（`download_models.py` 已自动清理）。

### 坑 2：声纹模型自己往网上重下

**原因**：`models/ecapa/hyperparams.yaml` 里 `pretrained_path` 写的是仓库名。
**修法**：`download_models.py` 会自动改成 `'.'`。若手工下载过模型，检查这一行。

### 坑 3：中文转写是繁体

**现象**：输出「今天天氣不錯」。
**修法**：`ears.py` 里已用 opencc 转简体；若没生效，装 `opencc-python-reimplemented`。

### 坑 4：`--json` 不是每个脚本都有

| 脚本 | 有 `--json` 吗 |
|---|---|
| `music.py` / `emotion.py` / `who.py` / `listen.py` | 有 |
| `ears.py` | **没有**（本来只输出 JSON） |

---

## 第 5 步：按需跑典型任务

```bash
# 转写（长音频慢，先切一段试）
python ears.py 音频.wav --model small

# 音乐特征（不需要任何模型）
python music.py 一首歌.wav --json

# 情绪（说话可用；唱歌不准，见 README）
python emotion.py 一段话.wav --json

# 认人：refs 目录里 名字.wav
python who.py 一段音频.wav --refs refs/ --json

# 综合
python listen.py 音频.wav [--refs refs/] [--asr] [--json]
```

**注意执行顺序**：`listen.py` 内部会调其它四个脚本，所以单独调某个脚本报错时，
先单独修那个脚本，不要直接怀疑 `listen.py`。

---

## 第 6 步：别做的事（重要）

1. **不要把 `models/` 提交进 Git**（.gitignore 已排；900 MB 且各有许可）
2. **不要把真人语音/声纹提交或公开**（声纹是生物识别信息，很多地区受法律保护）
3. **不要声称"情感识别对唱歌也准"**——模型是英语语音训练的，唱歌效果差，README 已注明
4. **不要把本项目说成"全部 MIT"**——代码是 MIT，模型各有许可（见 NOTICE.md）
5. **不要为了跑通而绕过 `--text-only`**：那是给群聊场景的隐私策略，不是 bug

---

## 排查顺序（报错时按这个顺序）

```
1. python --version / 用了哪个解释器
2. ffmpeg -version 能跑吗
3. models/ 三个目录都在吗（缺哪个就 download_models.py --only 那个）
4. 单独跑 music.py（不依赖模型）→ 排掉 librosa/scipy 问题
5. 单独跑 emotion.py / who.py → 排掉模型加载问题
6. 最后才跑 listen.py
```

`music.py` 是最干净的探针：它只依赖 librosa + ffmpeg，能跑通就说明基础环境没问题。

---

## 给 AI 的沟通建议

- 装完先**回报三个事实**：Python 版本、ffmpeg 是否有、模型目录是否齐
- **不要**只说"安装成功"，要说"用 demo 跑出了什么"（附 `summary` 一行）
- 用户话少时（"装一下"），按本文档全流程走完再报一句结论
