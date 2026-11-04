# -*- coding: utf-8 -*-
"""
一键下载本项目要用的三个模型到 ./models/（不发模型权重，只给下载脚本）。

  models/whisper-small     语音转文字（faster-whisper，CTranslate2 格式）
  models/ecapa             声纹 embedding（speechbrain ECAPA-TDNN）
  models/emotion_superb    情感 4 类（wav2vec2 SUPERB ER）

用法：
  python download_models.py              # 下全部
  python download_models.py --only ecapa # 只下某一个（whisper/ecapa/emotion）

国内网络提示：HuggingFace 直连不通时设 HF_ENDPOINT=https://hf-mirror.com
"""
import argparse
import os
import shutil
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.environ.get('LISTEN_MODELS', os.path.join(HERE, 'models'))
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

UA = {'User-Agent': 'huggingface-hub/0.25 (python-urllib)', 'Accept': '*/*'}


def dl(url, dst, retries=5):
    if os.path.exists(dst) and os.path.getsize(dst) > 1024:
        print(f'  skip {os.path.basename(dst)}')
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    for i in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=120) as r, open(dst + '.part', 'wb') as f:
                total = int(r.headers.get('content-length') or 0)
                got = 0
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    got += len(chunk)
            if total and got != total:
                raise IOError(f'长度不符 {got}/{total}')
            os.replace(dst + '.part', dst)
            print(f'  ok {os.path.basename(dst)} {os.path.getsize(dst) / 1048576:.1f} MB')
            return
        except Exception as e:
            print(f'  重试 {i}/{retries}: {type(e).__name__} {str(e)[:100]}')
            if os.path.exists(dst + '.part'):
                os.remove(dst + '.part')
    raise SystemExit(f'下载失败：{url}')


def get_whisper():
    print('[1/3] whisper-small')
    from faster_whisper import WhisperModel
    dst = os.path.join(MODELS, 'whisper-small')
    os.makedirs(dst, exist_ok=True)
    # faster-whisper 会用 HF 缓存；这里显式指定下载目录，避免污染全局缓存
    from huggingface_hub import snapshot_download
    snapshot_download('Systran/faster-whisper-small', local_dir=dst,
                      allow_patterns=['config.json', 'model.bin', 'tokenizer.json', 'vocabulary.txt'])
    print('  done')


def get_emotion():
    print('[2/3] emotion_superb')
    dst = os.path.join(MODELS, 'emotion_superb')
    base = 'https://hf-mirror.com/superb/wav2vec2-base-superb-er/resolve/main/'
    for f in ('config.json', 'preprocessor_config.json', 'pytorch_model.bin'):
        dl(base + f, os.path.join(dst, f))


def get_ecapa():
    print('[3/3] ecapa')
    dst = os.path.join(MODELS, 'ecapa')
    try:
        from huggingface_hub import snapshot_download
        snapshot_download('speechbrain/spkrec-ecapa-voxceleb', local_dir=dst)
    except Exception as e:
        print(f'  snapshot_download 失败（{type(e).__name__}），改逐文件下载')
        base = 'https://hf-mirror.com/speechbrain/spkrec-ecapa-voxceleb/resolve/main/'
        for f in ('hyperparams.yaml', 'embedding_model.ckpt', 'classifier.ckpt',
                  'mean_var_norm_emb.ckpt', 'label_encoder.txt', 'config.json'):
            dl(base + f, os.path.join(dst, f))

    # speechbrain 的两个坑，这里直接修好：
    # ① hyperparams.yaml 里写的是仓库名，会去网上重拉 → 改成相对本目录
    # ② Windows 上它默认用符号链接摆文件（普通账户没权限，WinError 1314）→ 清掉 .cache 标记
    hp = os.path.join(dst, 'hyperparams.yaml')
    if os.path.exists(hp):
        txt = open(hp, encoding='utf-8').read()
        txt = txt.replace('pretrained_path: speechbrain/spkrec-ecapa-voxceleb',
                          "pretrained_path: '.'")
        open(hp, 'w', encoding='utf-8').write(txt)
        print('  已把 hyperparams.yaml 的 pretrained_path 指向本地')
    cache = os.path.join(dst, '.cache')
    if os.path.isdir(cache):
        shutil.rmtree(cache, ignore_errors=True)
        print('  已清掉 .cache 标记（避免 hub 当缓存去建符号链接）')
    print('  done')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', choices=['whisper', 'ecapa', 'emotion'], help='只下某一个')
    a = ap.parse_args()

    print(f'模型目录：{MODELS}')
    os.makedirs(MODELS, exist_ok=True)
    if a.only:
        {'whisper': get_whisper, 'ecapa': get_ecapa, 'emotion': get_emotion}[a.only]()
    else:
        get_whisper()
        get_emotion()
        get_ecapa()
    print('\n全部完成。可以试：python listen.py 你的音频.wav --asr')
