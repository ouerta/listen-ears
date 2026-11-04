# -*- coding: utf-8 -*-
"""
听感情：这段（唱/说）是开心、生气、悲伤还是中性？
用法：python emotion.py <音频> [--json]

模型：superb/wav2vec2-base-superb-er（4 类，本地 models/emotion_superb）
做法：4 秒切窗 / 2 秒步长、按响度过滤 → 每个窗出概率 → 汇总成占比 + 时间轴
注意：模型是英语语音情感训练的，用在中文歌声上只能当粗略参考（README 里说明）。
"""
import argparse, json, os, subprocess, sys, tempfile

def _models_dir():
    """模型目录：优先环境变量 LISTEN_MODELS，否则用脚本旁边的 models/"""
    env = os.environ.get('LISTEN_MODELS', '').strip()
    if env:
        return env
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')



import numpy as np

SR = 16000
# 模型目录见 _models_dir()（可用 LISTEN_MODELS 指定）
MODEL_SUBDIR = 'emotion_superb'
WIN = 4.0
HOP = 2.0
MIN_RMS = 0.02
ZH = {'neu': '中性', 'hap': '开心', 'ang': '生气', 'sad': '悲伤',
      'neutral': '中性', 'happy': '开心', 'angry': '生气', 'sadness': '悲伤'}


def to_wav(src, dst):
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', src, '-ac', '1', '-ar', str(SR), '-f', 'wav', dst], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('audio')
    ap.add_argument('--json', action='store_true')
    a = ap.parse_args()

    import torch, librosa
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    model_dir = os.path.join(_models_dir(), MODEL_SUBDIR)
    if not os.path.isdir(model_dir):
        raise SystemExit(f'找不到情感模型目录：{model_dir}\n先运行 python download_models.py')
    fe = AutoFeatureExtractor.from_pretrained(model_dir)
    md = AutoModelForAudioClassification.from_pretrained(model_dir)
    md.eval()
    labels = {i: str(lab).lower() for i, lab in md.config.id2label.items()}

    wav = tempfile.mktemp(suffix='.wav')
    to_wav(a.audio, wav)
    y, _ = librosa.load(wav, sr=SR, mono=True)

    # 短音频自适应：语音消息常常只有 2–3 秒，4 秒窗口会一个都放不下（实测报"没有有声段"）
    win = int(min(WIN, max(1.0, len(y) / SR)) * SR)
    hop = max(int(SR * 0.5), int(win / 2))
    # ⚠️ 音频长度刚好等于窗口时，range(0, len-win) 是空的 → 至少保一窗（踩过）
    if len(y) <= win:
        starts = [0]
    else:
        starts = list(range(0, len(y) - win + 1, hop))
    rms_all = [float(np.sqrt(np.mean(y[s:s + win] ** 2))) for s in starts]
    pos = sorted([r for r in rms_all if r > 0], reverse=True)
    floor = max(MIN_RMS, 0.4 * pos[len(pos) // 2] if pos else 0.0)

    segs = []
    for idx, s in enumerate(starts):
        if rms_all[idx] < floor:
            continue
        chunk = y[s:s + win]
        inputs = fe(chunk, sampling_rate=SR, return_tensors='pt')
        with torch.no_grad():
            logits = md(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        segs.append({
            'start': round(s / SR, 2),
            'end': round((s + win) / SR, 2),
            'top': labels[int(probs.argmax())],
            'probs': {labels[i]: round(float(probs[i]), 3) for i in range(len(probs))},
        })

    if not segs:
        print(json.dumps({'file': a.audio, 'error': '没有有声段'}, ensure_ascii=False))
        return

    # 汇总：按窗口数加权（每个窗 4 秒）
    tally, prob_sum = {}, {}
    for s in segs:
        tally[s['top']] = tally.get(s['top'], 0) + 1
        for k, v in s['probs'].items():
            prob_sum[k] = prob_sum.get(k, 0.0) + v
    n = len(segs)
    dist = {ZH.get(k, k): round(v / n, 3) for k, v in prob_sum.items()}
    order = sorted(dist.items(), key=lambda x: -x[1])
    result = {
        'file': a.audio,
        'seconds_analyzed': round(n * WIN, 1),
        'windows': n,
        'dominant': order[0][0],
        'distribution': dict(order),
        'top_counts': {ZH.get(k, k): v for k, v in sorted(tally.items(), key=lambda x: -x[1])},
        'timeline': [{'s': s['start'], 'e': s['end'], 'e_': ZH.get(s['top'], s['top'])} for s in segs],
    }
    print(json.dumps(result, ensure_ascii=False) if a.json else json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    sys.exit(main())
