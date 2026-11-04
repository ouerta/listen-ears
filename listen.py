# -*- coding: utf-8 -*-
"""
综合听音：一次跑完「说了什么 / 什么调 / 谁在唱 / 什么情绪」
用法：python listen.py <音频> [--refs 参考目录] [--asr] [--json]

各层分别调用：
  music.py   调性/速度/音高/音色/响度（纯计算，快）
  emotion.py 情感 4 类（wav2vec2，说话准、唱歌仅参考）
  who.py     声纹认人（有 refs 才跑）
  ears.py    语音转文字（--asr 才跑，慢）
"""
import argparse, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def run(script, args):
    cmd = [PY, os.path.join(HERE, script)] + args
    try:
        env = {**os.environ, 'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8'}
        out = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8',
                               errors='replace', timeout=1800, env=env)
        txt = (out.stdout or '').strip()
        if txt.startswith('{'):
            return json.loads(txt)
        return {'error': f'{script} 输出不是 JSON', 'stderr': (out.stderr or '')[-300:]}
    except Exception as e:
        return {'error': f'{script} 执行失败: {type(e).__name__}: {e}'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('audio')
    ap.add_argument('--refs', default='')
    ap.add_argument('--asr', action='store_true', help='同时做语音转文字（慢）')
    ap.add_argument('--text-only', action='store_true', dest='text_only',
                    help='只做语音转文字（群聊场景：不返回情感/声纹/音乐特征）')
    ap.add_argument('--json', action='store_true')
    a = ap.parse_args()

    report = {'file': a.audio}

    # text-only：只转写。群聊里别人发的语音，只给"他说了什么"，不给情感/声纹这些更私密的东西。
    if a.text_only:
        report['speech'] = run('ears.py', [a.audio, '--model', 'small'])
        sp = report.get('speech') or {}
        report['duration'] = sp.get('duration')
        report['summary'] = ('语音转写：' + sp['text']) if sp.get('text') else '（没听清）'
        print(json.dumps(report, ensure_ascii=False) if a.json else json.dumps(report, ensure_ascii=False, indent=2))
        return

    report['music'] = run('music.py', [a.audio, '--json'])
    report['emotion'] = run('emotion.py', [a.audio, '--json'])
    if a.refs:
        report['who'] = run('who.py', [a.audio, '--refs', a.refs, '--json'])
    if a.asr:
        report['speech'] = run('ears.py', [a.audio, '--model', 'small'])

    # 汇总一句人话
    lines = []
    m = report.get('music') or {}
    if m.get('key'):
        lines.append('调性 %s（%s）' % (m['key'].get('estimate'), '可信' if m['key'].get('reliable') else '置信不足'))
    if m.get('tempo_bpm'):
        lines.append('速度约 %.0f BPM' % m['tempo_bpm'])
    if (m.get('pitch') or {}).get('median_hz'):
        p = m['pitch']
        lines.append('音高中位 %.0f Hz（%.1f 个八度跨度 %.1f 半音，起伏 %.1f）'
                     % (p['median_hz'], 1 + p['range_semitones'] / 12, p['range_semitones'], p['variability_st']))
    if (m.get('timbre') or {}).get('centroid_hz'):
        lines.append('音色质心 %.0f Hz' % m['timbre']['centroid_hz'])
    if m.get('summary'):
        lines.append('音乐侧判断：' + m['summary'])
        # 说话片段也算出“调性/速度”，但意义不大；明确标注，免得误导。
        if m.get('key') and not m['key'].get('reliable'):
            lines.append('（调性置信不足，这段可能更像说话而不是音乐，调性/速度仅供参考）')

    e = report.get('emotion') or {}
    if e.get('dominant'):
        dist = '、'.join('%s %.0f%%' % (k, v * 100) for k, v in e['distribution'].items())
        lines.append('情感：主导 %s（%s）' % (e['dominant'], dist))

    w = report.get('who') or {}
    if w.get('speakers'):
        lines.append('音色：' + '、'.join(
            ('%s %.0f 秒' % (x.get('who') or ('簇%s' % x.get('cluster')), x.get('seconds', 0))) for x in w['speakers'][:4]))

    s = report.get('speech') or {}
    if s.get('text'):
        lines.append('说了：' + s['text'][:200])

    report['summary'] = '；'.join(lines)
    print(json.dumps(report, ensure_ascii=False) if a.json else json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    sys.exit(main())
