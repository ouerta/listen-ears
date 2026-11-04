# -*- coding: utf-8 -*-
"""
听音乐：调性 / 速度 / 音高 / 音色 / 响度 + 一句人话总结
用法：python music.py <音频> [--json]
思路：librosa 算特征（chroma 估调性、pyin 取值、MFCC+谱特征算音色、beat 算速度），
     最后用可解释的阈值拼一句结论（不做训练、不下载模型）。
"""
import argparse, json, subprocess, sys, tempfile
import numpy as np

# ⚠️ 兼容补丁：新版 scipy 把 signal.hann 移到了 signal.windows.hann，
#    而这里装的 librosa 还按老名字调用 → beat_track 会 AttributeError。
import scipy.signal as _ss
if not hasattr(_ss, 'hann'):
    _ss.hann = _ss.windows.hann

SR = 22050
# Krumhansl-Schmuckler 调性模板（大调/小调）
MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def to_wav(src, dst):
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', src, '-ac', '1', '-ar', str(SR),
                    '-f', 'wav', dst], check=True)


def estimate_key(chroma):
    prof = chroma.mean(axis=1)
    best = (-1, None, None)
    for i in range(12):
        for name, tmpl in (('大调', MAJOR), ('小调', MINOR)):
            r = np.corrcoef(np.roll(prof, -i), tmpl)[0, 1]
            if r > best[0]:
                best = (r, NAMES[i] + ' ' + name, i)
    return best[0], best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('audio')
    ap.add_argument('--json', action='store_true', help='只输出 JSON')
    a = ap.parse_args()

    wav = tempfile.mktemp(suffix='.wav')
    to_wav(a.audio, wav)

    import librosa
    y, sr = librosa.load(wav, sr=SR, mono=True)
    dur = len(y) / sr

    # ── 调性 ──
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    key_r, key_name = estimate_key(chroma)

    # ── 速度（自己算，不依赖 beat_track）──
    # onset 强度包络 → 自相关 → 在 60–200 BPM 里找峰
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    fps = sr / 512
    ac = librosa.autocorrelate(onset_env, max_size=len(onset_env) // 2)
    lo = max(1, int(fps * 60 / 200))
    hi = min(len(ac) - 1, int(fps * 60 / 60))
    lag = lo + int(np.argmax(ac[lo:hi])) if hi > lo else lo
    tempo = 60.0 * fps / max(1, lag)

    # ── 音高（人声/主旋律范围）──
    # 关键：整轨混音里鼓/噪声会让 pyin 报出大量错音（实测「雨爱」均值 373Hz、跨度 43 半音），
    # 所以①收窄到 80–600Hz；②用 voiced_prob > 0.5 过滤低置信帧；③半音差以中位数为基准。
    f0, vflag, vprob = librosa.pyin(y, fmin=80, fmax=600, sr=sr, frame_length=2048)
    good = (~np.isnan(f0)) & (vprob > 0.5)
    pv = f0[good]
    if len(pv) > 20:
        semis = 12 * np.log2(pv / np.median(pv))
        # 抗离群的起伏度：MAD（中位数绝对偏差）。混音里的八度跳变会把标准差拉到 9 个半音
        # （实测《稻香》std=9.44 → 误判成"激烈"），MAD 基本不受这种离群值影响。
        mad = float(np.median(np.abs(semis - np.median(semis))))
        # 逐帧音高变化：剔掉 >6 半音的跳变（八度误判），再取稳健离散度。
        # 这比整体标准差/MAD 靠谱——后者被八度跳变和旋律走向一起放大。
        d = np.diff(semis)
        d = d[np.abs(d) < 6]
        frame_move = float(1.4826 * np.median(np.abs(d - np.median(d)))) if len(d) > 10 else 0.0
        pitch = {
            'voiced_ratio': round(float(np.sum(good) / max(1, len(f0))), 3),
            'frames_used': int(len(pv)),
            'mean_hz': round(float(np.mean(pv)), 1),
            'median_hz': round(float(np.median(pv)), 1),
            'low_hz': round(float(np.percentile(pv, 5)), 1),
            'high_hz': round(float(np.percentile(pv, 95)), 1),
            'range_semitones': round(float(np.percentile(semis, 95) - np.percentile(semis, 5)), 2),
            'variability_st': round(float(np.std(semis)), 2),
            'variability_mad_st': round(1.4826 * mad, 2),   # 稳健起伏度
            'frame_move_st': round(frame_move, 3),          # 逐帧起伏（情绪判断主用）
        }
    else:
        pitch = {'voiced_ratio': 0.0, 'frames_used': int(len(pv)), 'mean_hz': None, 'median_hz': None,
                 'low_hz': None, 'high_hz': None, 'range_semitones': None, 'variability_st': None,
                 'variability_mad_st': None, 'frame_move_st': None}

    # ── 音色（谱特征 + MFCC）──
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    roll = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
    flat = librosa.feature.spectral_flatness(y=y)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    timbre = {
        'centroid_hz': round(float(np.mean(cent)), 0),
        'bandwidth_hz': round(float(np.mean(bw)), 0),
        'rolloff85_hz': round(float(np.mean(roll)), 0),
        'flatness': round(float(np.mean(flat)), 4),
        'mfcc_std_mean': round(float(np.mean(np.std(mfcc, axis=1))), 2),
        'timbre_change': round(float(np.mean(np.std(mfcc[:, ::5], axis=1))), 2),
    }

    # ── 响度/动态 ──
    # 用「相对峰值」的 dB，并把底噪截到 -60 dB；
    # 之前直接 20*log10(rms) 会让静音段产生 -180dB，动态范围算出 149 dB 这种假数字。
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    peak = float(np.max(rms)) or 1e-8
    db = 20 * np.log10(np.maximum(rms, 1e-8) / peak)
    db = np.maximum(db, -60.0)
    loud = {
        'rel_db_mean': round(float(np.mean(db)), 1),
        'dynamic_range_db': round(float(np.percentile(db, 95) - np.percentile(db, 10)), 1),
    }

    # ── 一句人话 ──
    tags = []
    # 混音太杂时 pyin 基本量不出东西（实测「雨爱」DJ 版只剩 43 个有效帧），
    # 这种情况直接提示先做 人声分离，而不是硬给结论。
    low_voice = pitch['voiced_ratio'] < 0.15
    if low_voice:
        tags.append('⚠ 人声占比很低（%.2f），鼓/伴奏太多，音高不可信 → 建议先用 MSST 分离人声再分析' % pitch['voiced_ratio'])
    if pitch['mean_hz']:
        f = pitch['mean_hz']
        tags.append('偏低沉的男声区' if f < 150 else '中音区' if f < 260 else '偏高的女声区')
        if pitch['range_semitones'] and pitch['range_semitones'] > 12:
            tags.append('音域跨度大')
        if pitch['variability_st'] and pitch['variability_st'] > 2.2:
            tags.append('音高起伏明显（情绪外放）')
        else:
            tags.append('音高平稳')
    c = timbre['centroid_hz']
    tags.append('音色偏亮' if c > 2600 else '音色偏暗/浑厚' if c < 1400 else '音色中性')
    if timbre['flatness'] > 0.08:
        tags.append('噪声成分多（可能有鼓/气声）')
    tags.append('节奏偏快' if tempo >= 110 else '节奏偏慢' if tempo <= 80 else '中等速度')
    if loud['dynamic_range_db'] > 14:
        tags.append('动态大（强弱对比明显）')

    # ── 情绪（规则版，不靠模型）──
    # 用三个可解释的维度打分，再映射到标签。唱歌用模型判情绪不可信（英语语音模型），
    # 所以这一层专门给音乐用：能量 / 明亮度 / 张力。
    def clamp01(x):
        return max(0.0, min(1.0, float(x)))

    is_minor = '小调' in key_name
    # 只用**可靠**特征判情绪：速度（稳）、调性（置信>0.5 才算）、音色亮度（稳）、动态范围（稳）。
    # 音高起伏即使换成 MAD / 逐帧变化，在混音上仍然被八度跳变污染（实测《稻香》std 9.44、
    # MAD 10.23），所以它**只报数、不参与判断**。
    pitch_ok = (pitch['voiced_ratio'] or 0) >= 0.15
    key_ok = key_r > 0.5

    brightness = clamp01((timbre['centroid_hz'] - 800) / 3200)
    energy = 0.7 * clamp01((tempo - 60) / 90) + 0.3 * clamp01(loud['dynamic_range_db'] / 20)

    reasons = []
    if tempo >= 125:
        reasons.append('速度 %.0f BPM 很快' % tempo)
    elif tempo >= 100:
        reasons.append('速度 %.0f BPM 中等偏快' % tempo)
    elif tempo <= 85:
        reasons.append('速度 %.0f BPM 偏慢' % tempo)
    else:
        reasons.append('速度 %.0f BPM 中等' % tempo)
    if key_ok:
        reasons.append('小调（偏暗/偏忧）' if is_minor else '大调（偏明亮）')
    else:
        reasons.append('调性置信不足，未参与判断')
    reasons.append('音色偏亮' if timbre['centroid_hz'] > 2600 else
                   '音色偏暗' if timbre['centroid_hz'] < 1400 else '音色中性')
    if loud['dynamic_range_db'] > 14:
        reasons.append('强弱对比明显')
    if not pitch_ok:
        reasons.append('人声太少，音高只作参考')

    if tempo >= 125 and brightness >= 0.35:
        mood = '激烈/热血'
    elif tempo >= 100 and key_ok and not is_minor:
        mood = '欢快/明亮'
    elif tempo <= 90 and key_ok and is_minor:
        mood = '悲伤/低沉'
    elif tempo <= 95:
        mood = '平静/温情'
    else:
        mood = '抒情/中速'

    result = {
        'file': a.audio,
        'duration': round(dur, 2),
        'key': {'estimate': key_name, 'confidence': round(float(key_r), 3),
                'reliable': bool(key_r > 0.5)},
        'tempo_bpm': round(tempo, 1),
        'mood': {
            'label': mood,
            'confidence': 'high' if (pitch_ok and key_ok) else 'low',
            'dims': {'energy': round(energy, 2), 'brightness': round(brightness, 2)},
            'reasons': reasons,
        },
        'tempo_hint': ('检测值偏快，也可能是一半的 %.1f BPM' % (tempo / 2)) if tempo > 140 else '',
        # 同一首实测：人声版 129.2、混音版 64.6 —— 正好差一倍。倍速歧义没法完全消除，
        # 所以把候选都列出来，由使用者/AI 结合听感判断。
        'tempo_candidates': [round(t, 1) for t in (tempo / 2, tempo, tempo * 2) if 50 <= t <= 200],
        'pitch': pitch,
        'timbre': timbre,
        'loudness': loud,
        'summary': '、'.join(['【' + mood + '】'] + tags),
    }

    if a.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    sys.exit(main())
