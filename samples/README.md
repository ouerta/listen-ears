# 样本目录

这里**不放任何真人声音**（声纹属于生物特征，不适合随项目分发）。

自己准备测试素材：

1. **说话**：手机录一段 5–10 秒，或直接用系统 TTS 生成
   ```powershell
   Add-Type -AssemblyName System.Speech
   $s = New-Object System.Speech.Synthesis.SpeechSynthesizer
   $s.SetOutputToWaveFile("samples\tts.wav")
   $s.Speak("今天天气不错，我下午准备去图书馆看一会儿书。")
   $s.Dispose()
   ```
2. **音乐**：任意 mp3/wav。**建议先做人声分离再分析**（见 README 的坑 #1）
3. **认人**：在 `refs/` 下按 `名字.wav` 放参考样本，然后 `python who.py 音频.wav --refs refs/`