# -*- coding: utf-8 -*-
"""
认人：这段音频里有几个音色？分别是谁？
用法：
  python who.py <音频>                      # 只聚类，看有几个不同音色
  python who.py <音频> --refs <参考目录>     # 跟参考样本比对，直接报名字
参考目录里的文件用「名字.wav / 名字.mp3」命名，就按文件名当名字。

技术：ECAPA-TDNN 声纹 embedding（speechbrain spkrec-ecapa-voxceleb，本地 models/ecapa）
      → 每 2.5 秒切窗、跳过安静段 → 有参考就余弦比对，没参考就 KMeans 聚类
"""
import argparse, json, os, subprocess, sys, tempfile

def _models_dir():
    """模型目录：优先环境变量 LISTEN_MODELS，否则用脚本旁边的 models/"""
    env = os.environ.get('LISTEN_MODELS', '').strip()
    if env:
        return env
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')



import numpy as np

# ── Windows 兼容补丁（踩过）──
# speechbrain 在 Windows 上默认用「符号链接」策略去摆 pretrained 文件，普通账户没这个权限，
# 会抛 WinError 1314。这里把 symlink_to 兜底成复制。
import pathlib, shutil
_orig_symlink = pathlib.Path.symlink_to
def _safe_symlink(self, target, target_is_directory=False):
    try:
        _orig_symlink(self, target, target_is_directory)
    except OSError:
        t = pathlib.Path(target)
        if t.is_dir() or target_is_directory:
            shutil.copytree(t, self, dirs_exist_ok=True)
        else:
            self.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(t, self)
pathlib.Path.symlink_to = _safe_symlink
os.environ.setdefault('HF_HUB_OFFLINE', '1')

SR = 16000
# 注意用 ecapa_clean：从 HF 下到 local_dir 的目录里带 .cache 标记，hub 会当成缓存去建链接；
# 而且 hyperparams.yaml 里写死了仓库名，得改成这个本地路径（已改）。
MODEL_SUBDIR = 'ecapa'
WIN = 2.5      # 每段时长（秒）
HOP = 1.25     # 步长（秒）
# ⚠️ 音量门限很关键：静音窗会让 ECAPA 输出乱七八糟的向量（实测同人相似度会掉到 -0.002）。
#    所以既设绝对下限，也按整段响度的中位数做相对过滤。
MIN_RMS = 0.02
SIM_THRESHOLD = 0.18   # 实测同人 0.26–0.79、不同人 0.02–0.15；取 0.18 减少「未知」误判


def to_wav16k(src, dst):
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', src, '-ac', '1', '-ar', str(SR), '-f', 'wav', dst], check=True)


def load(path):
    import librosa
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('audio')
    ap.add_argument('--refs', default='', help='参考样本目录（文件名即名字）')
    ap.add_argument('--threshold', type=float, default=SIM_THRESHOLD)
    ap.add_argument('--json', action='store_true')
    a = ap.parse_args()

    import torch
    from speechbrain.inference.speaker import EncoderClassifier
    model_dir = os.path.join(_models_dir(), MODEL_SUBDIR)
    if not os.path.isdir(model_dir):
        raise SystemExit(f'找不到声纹模型目录：{model_dir}\n先运行 python download_models.py')
    clf = EncoderClassifier.from_hparams(source=model_dir, savedir=model_dir, run_opts={'device': 'cpu'})

    def embed(y):
        t = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            e = clf.encode_batch(t).squeeze().cpu().numpy()
        return e / (np.linalg.norm(e) + 1e-9)

    # ── 参考样本：只取有声片段求平均（整段 30 秒里静音会把向量带偏）──
    refs = []
    if a.refs and os.path.isdir(a.refs):
        for fn in sorted(os.listdir(a.refs)):
            if not fn.lower().endswith(('.wav', '.mp3', '.flac', '.m4a', '.ogg')):
                continue
            name = os.path.splitext(fn)[0]
            wav = tempfile.mktemp(suffix='.wav')
            try:
                to_wav16k(os.path.join(a.refs, fn), wav)
                y = load(wav)
                w = int(WIN * SR)
                valid = [y[i:i + w] for i in range(0, max(0, len(y) - w), w)
                         if float(np.sqrt(np.mean(y[i:i + w] ** 2))) > MIN_RMS]
                if not valid:
                    print(f'[refs] 跳过 {fn}：没有足够响的片段', file=sys.stderr)
                    continue
                # 最多用 8 个最响片段
                valid.sort(key=lambda c: -float(np.sqrt(np.mean(c ** 2))))
                embs_ref = np.stack([embed(c) for c in valid[:8]])
                v = embs_ref.mean(axis=0)
                refs.append((name, v / (np.linalg.norm(v) + 1e-9)))
            except Exception as e:
                print(f'[refs] 跳过 {fn}: {e}', file=sys.stderr)

    # ── 主音频 ──
    wav = tempfile.mktemp(suffix='.wav')
    to_wav16k(a.audio, wav)
    y = load(wav)
    win, hop = int(WIN * SR), int(HOP * SR)
    segs = []
    import librosa
    # 先算一遍所有窗的响度，取「相对门限」= 前 50% 窗的中位数 * 0.4，避免整轨偏轻时全被丢掉
    rms_all = []
    for start in range(0, max(0, len(y) - win), hop):
        chunk = y[start:start + win]
        rms_all.append(float(np.sqrt(np.mean(chunk ** 2))))
    pos = sorted([r for r in rms_all if r > 0], reverse=True)
    rel_floor = 0.4 * pos[len(pos) // 2] if pos else 0.0
    floor = max(MIN_RMS, rel_floor)

    for idx, start in enumerate(range(0, max(0, len(y) - win), hop)):
        if rms_all[idx] < floor:
            continue
        chunk = y[start:start + win]
        segs.append((round(start / SR, 2), round((start + win) / SR, 2), embed(chunk)))

    if not segs:
        print(json.dumps({'file': a.audio, 'error': '没有有声段（全静音或太轻）'}, ensure_ascii=False))
        return

    embs = np.stack([s[2] for s in segs])

    # ── 有参考：按余弦相似度认人 ──
    if refs:
        names = [r[0] for r in refs]
        R = np.stack([r[1] for r in refs])
        sims = embs @ R.T                       # (n_seg, n_ref)
        out_segs = []
        for i, (st, en, _) in enumerate(segs):
            j = int(np.argmax(sims[i]))
            best = float(sims[i][j])
            who = names[j] if best >= a.threshold else '未知'
            out_segs.append({'start': st, 'end': en, 'who': who, 'sim': round(best, 3)})
        tally = {}
        for s in out_segs:
            tally[s['who']] = tally.get(s['who'], 0) + (s['end'] - s['start'])
        result = {
            'file': a.audio,
            'mode': 'identify',
            'refs': names,
            'segments': out_segs,
            'speakers': [{'who': k, 'seconds': round(v, 1)} for k, v in sorted(tally.items(), key=lambda x: -x[1])],
        }
    else:
        # ── 无参考：KMeans 聚类，用轮廓系数挑 k ──
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        best = (None, -2, 1)
        for k in range(1, min(5, len(segs))):
            km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(embs)
            sc = -1.0 if k == 1 else silhouette_score(embs, km.labels_)
            if sc > best[1]:
                best = (km, sc, k)
        km, sc, k = best
        # 单一歌手的 embedding 方差也会让 KMeans 硬分出好几簇（实测 3 簇、轮廓系数 0.124），
        # 所以轮廓系数不够高就当作「只有 1 个音色」。
        if k > 1 and sc < 0.25:
            k = 1
        labels = list(km.labels_) if k > 1 else [0] * len(segs)
        # 时序平滑：3 窗滑动取众数，避免簇标签在相邻窗口来回跳
        if k > 1:
            sm = []
            for i in range(len(labels)):
                win3 = labels[max(0, i - 1): i + 2]
                sm.append(max(set(win3), key=win3.count))
            labels = sm
        out_segs = [{'start': s[0], 'end': s[1], 'cluster': int(labels[i])} for i, s in enumerate(segs)]
        tally = {}
        for s in out_segs:
            tally[s['cluster']] = tally.get(s['cluster'], 0) + (s['end'] - s['start'])
        result = {
            'file': a.audio,
            'mode': 'cluster',
            'clusters': k,
            'silhouette': round(float(sc), 3),
            'segments': out_segs,
            'speakers': [{'cluster': c, 'seconds': round(v, 1)} for c, v in sorted(tally.items(), key=lambda x: -x[1])],
        }

    result['window'] = {'sec': WIN, 'hop': HOP, 'used': len(segs)}
    print(json.dumps(result, ensure_ascii=False) if a.json else json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    sys.exit(main())
