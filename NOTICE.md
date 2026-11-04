# 第三方组件与许可（NOTICE）

本项目（listen-ears）的**自身代码**以 MIT 许可发布。运行时会用到下列第三方组件/模型，
它们的许可与出处如下。分发或再发布时请保留本文件。

## 1. 语音转写

- **faster-whisper** — MIT — https://github.com/SYSTRAN/faster-whisper
- **Whisper 模型**（`Systran/faster-whisper-small`，OpenAI Whisper 的 CTranslate2 转换版）— MIT — https://github.com/openai/whisper

## 2. 声纹（说话人识别）

- **SpeechBrain** — Apache-2.0 — https://github.com/speechbrain/speechbrain
- **spkrec-ecapa-voxceleb**（ECAPA-TDNN，VoxCeleb 训练）— Apache-2.0 — https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb
  - 训练数据 VoxCeleb 的使用条款见 https://www.robots.ox.ac.uk/~vgg/data/voxceleb/

## 3. 情感识别

- **SUPERB** 基准 — https://superbenchmark.org/
- **wav2vec2-base-superb-er** — Apache-2.0 — https://huggingface.co/superb/wav2vec2-base-superb-er
  - 底座 wav2vec2 由 Facebook/Meta 发布（MIT / Apache-2.0，见模型卡）
  - **注意**：该模型在英语语音情感数据上训练，对中文歌声效果差，README 已注明。

## 4. 音频特征与工具

- **librosa** — ISC — https://librosa.org/
- **soundfile**（libsndfile）— BSD-3-Clause
- **NumPy** — BSD-3-Clause
- **SciPy** — BSD-3-Clause
- **scikit-learn**（仅无参考聚类时用）— BSD-3-Clause
- **OpenCC / opencc-python-reimplemented**（简繁转换）— Apache-2.0
- **ffmpeg** — LGPL-2.1+ / GPL（取决于编译选项）— https://ffmpeg.org/legal.html
  - 本项目**只通过命令行调用**系统已安装的 ffmpeg，不分发其二进制。

## 5. 本项目不包含的东西

- **不分发任何模型权重**（`download_models.py` 只从官方/镜像地址下载，使用时请遵守各自许可）
- **不包含任何真人声音样本**（`refs/` 与 `samples/` 由使用者自己准备）

## 6. 使用他人语音的合规提醒

`who.py` 处理的是**声纹**——在很多司法辖区属于生物识别信息。请：

1. 只在**获得对方知情同意**的前提下分析他人语音；
2. 不要公开、上传他人声音样本或声纹向量；
3. 商用前确认所在地区对生物识别数据的法规（如中国的《个人信息保护法》、欧盟 GDPR 等）。
