"""產生極性／相位測試音檔：粉紅雜訊分 8 段，每段 8 秒、段間 1 秒靜音。
用途：播給 Decibel Phase Scope（或日後 Pi 的 PHASE 畫面）看各種狀況長什麼樣子。
左聲道 = A 麥克風、右聲道 = B 麥克風。

    py -3 tools/make_phase_test.py            # 輸出 test_audio/phase_test.wav，並印出各段實測 correlation
"""
import os
import wave

import numpy as np

SR, SEG_S, GAP_S, RMS_DBFS = 48000, 8.0, 1.0, -20.0
rng = np.random.default_rng(2026)


def pink(n):
    # ponytail: FFT 整形的粉紅雜訊（1/f 功率），測試訊號用夠了
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / SR)
    spec[1:] /= np.sqrt(f[1:])
    spec[0] = 0
    spec[f < 40] = 0                      # 去掉極低頻，免得喇叭和點雲被拉偏
    x = np.fft.irfft(spec, n)
    return x / np.sqrt(np.mean(x ** 2))   # RMS = 1


def delay(x, ms):
    d = round(SR * ms / 1000)
    return np.concatenate([np.zeros(d), x[:-d]])


N = int(SR * SEG_S)
SEGMENTS = [   # (名稱, 預期看到什麼, 產生函式 → (L, R))
    ("1 同相（單聲道）",       "直立一條線，+1",               lambda a, b: (a, a)),
    ("2 極性接反",             "橫躺一條線，-1",               lambda a, b: (a, -a)),
    ("3 只有左聲道",           "往左上斜的線",                 lambda a, b: (a, 0 * a)),
    ("4 只有右聲道",           "往右上斜的線",                 lambda a, b: (0 * a, a)),
    ("5 完全不相關",           "圓形一團，約 0",               lambda a, b: (a, b)),
    ("6 一般立體聲（部分相關）", "直立的橢圓，約 +0.8",          lambda a, b: (a + 0.5 * b, a + 0.5 * pink(N))),
    ("7 梳狀濾波（B 晚 1 ms）", "鬆散一團，分頻正負交錯",        lambda a, b: (a, delay(a, 1.0))),
    ("8 正常擺位（B 只收漏音）", "幾乎只在左上斜線附近，約 0~+0.3", lambda a, b: (a, 0.3 * delay(a, 1.5) + 0.1 * b)),
]


def corr(l, r):
    den = np.sqrt(np.sum(l * l) * np.sum(r * r))
    return float(np.sum(l * r) / den) if den > 0 else 0.0


def build():
    fade = np.ones(N)
    k = int(SR * 0.02)                    # 20 ms 淡入淡出，避免段落交界爆音
    fade[:k] = np.linspace(0, 1, k)
    fade[-k:] = np.linspace(1, 0, k)
    gap = np.zeros((int(SR * GAP_S), 2))
    gain = 10 ** (RMS_DBFS / 20)
    parts, report, t = [], [], 0.0
    for name, expect, fn in SEGMENTS:
        l, r = fn(pink(N), pink(N))
        seg = np.stack([l * fade, r * fade], axis=1) * gain
        report.append((t, name, expect, corr(l, r)))
        parts += [seg, gap]
        t += SEG_S + GAP_S
    out = np.concatenate(parts)
    peak = np.max(np.abs(out))
    if peak > 0.89:                       # 保留約 -1 dBFS 餘裕
        out *= 0.89 / peak
    return out, report


def write_wav(path, x):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


if __name__ == "__main__":
    out, report = build()
    c = {name[0]: v for _, name, _, v in report}
    assert c["1"] > 0.999 and c["2"] < -0.999, c          # 同相／反相一定要是 ±1
    assert abs(c["5"]) < 0.05 and 0.7 < c["6"] < 0.9, c    # 不相關 ≈ 0、立體聲 ≈ 0.8
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test_audio", "phase_test.wav")
    write_wav(os.path.normpath(path), out)
    print(os.path.normpath(path))
    for t, name, expect, v in report:
        print(f"{int(t) // 60}:{int(t) % 60:02d}  {name:<16} corr {v:+.2f}  {expect}")
