# -*- coding: utf-8 -*-
"""
给 AI 装耳朵：一段音频 → 文字（带时间轴）+ 音高 + 停顿
用法：python ears.py <音频> [--model small] [--lang zh]
思路（用户给的）：ffmpeg 转 16k 单声道 → faster-whisper 出时间轴 → librosa 量音高和停顿
"""
import argparse, json, os, subprocess, sys, tempfile

def _models_dir():
    """模型目录：优先环境变量 LISTEN_MODELS，否则用脚本旁边的 models/"""
    env = os.environ.get('LISTEN_MODELS', '').strip()
    if env:
        return env
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')


import numpy as np


def to_wav16k(src, dst):
    subprocess.run(
        ['ffmpeg', '-v', 'error', '-y', '-i', src, '-ac', '1', '-ar', '16000', '-f', 'wav', dst],
        check=True
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('audio')
    ap.add_argument('--model', default='small')
    ap.add_argument('--lang', default='zh')
    ap.add_argument('--device', default='cpu')
    a = ap.parse_args()

    wav = tempfile.mktemp(suffix='.wav')
    to_wav16k(a.audio, wav)

    # ── 1. 说话内容 + 时间轴 ──
    from faster_whisper import WhisperModel
    # 优先用本地 models/whisper-small（download_models.py 下的），否则按名字让 faster-whisper 自己找
    local = os.path.join(_models_dir(), 'whisper-small')
    model_ref = local if os.path.isdir(local) else a.model
    model = WhisperModel(model_ref, device=a.device, compute_type='int8')
    segs, info = model.transcribe(wav, language=a.lang, vad_filter=True)
    segments = [{'start': round(s.start, 2), 'end': round(s.end, 2), 'text': s.text.strip()} for s in segs]

    # Whisper 常输出繁体（实测："今天天氣不錯"），统一转简体再返回，免得下游看着别扭
    try:
        from opencc import OpenCC
        _cc = OpenCC('t2s')
        for seg in segments:
            seg['text'] = _cc.convert(seg['text'])
    except Exception:
        pass

    # ── 2. 音高（人声基频）与停顿 ──
    import librosa
    y, sr = librosa.load(wav, sr=16000, mono=True)
    f0, voiced, _ = librosa.pyin(y, fmin=70, fmax=500, sr=sr)
    pitch_vals = f0[~np.isnan(f0)]
    rms = librosa.feature.rms(y=y, frame_length=512, hop_length=160)[0]
    hop_s = 160 / sr
    quiet_thr = max(0.008, float(np.median(rms)) * 0.25)

    pauses, i = [], 0
    while i < len(rms):
        if rms[i] < quiet_thr:
            j = i
            while j < len(rms) and rms[j] < quiet_thr:
                j += 1
            if (j - i) * hop_s >= 0.35:
                pauses.append([round(i * hop_s, 2), round(j * hop_s, 2)])
            i = j
        else:
            i += 1

    print(json.dumps({
        'duration': round(len(y) / sr, 2),
        'text': ''.join(s['text'] for s in segments),
        'segments': segments,
        'pitch': {
            'voiced_ratio': round(float(len(pitch_vals)) / max(1, len(f0)), 2),
            'mean_hz': round(float(np.mean(pitch_vals)), 1) if len(pitch_vals) else None,
            'low_hz': round(float(np.percentile(pitch_vals, 5)), 1) if len(pitch_vals) else None,
            'high_hz': round(float(np.percentile(pitch_vals, 95)), 1) if len(pitch_vals) else None,
        },
        'pauses': pauses[:30],
        'pause_count': len(pauses),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
