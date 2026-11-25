"""PHASE 畫面自我檢查：把 test_audio/phase_test.wav（tools/make_phase_test.py 產生）逐段餵給 compute_phase()，
數值不符預期就 assert 失敗；給輸出目錄時，每段兩頁（GONIO／BAND）各存一張 PNG，拿去跟 Decibel 截圖對照。

    python tools/check_phase.py [輸出目錄]      # 需 numpy、pygame；筆電沒有 sounddevice 等模組時自動塞空殼
"""
import os, sys, types, wave
os.environ["SDL_VIDEODRIVER"] = "dummy"; os.environ["SDL_AUDIODRIVER"] = "dummy"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
try:
    import sounddevice, gpiozero, lufs_core, scipy.signal  # noqa: F401  Pi 上都有
except ImportError:   # ponytail: 筆電沒有這些模組，塞空殼只為了能 import
    for name in ("sounddevice", "gpiozero", "lufs_core", "scipy", "scipy.signal"):
        sys.modules[name] = types.ModuleType(name)
    sys.modules["gpiozero"].Button = object
    sys.modules["lufs_core"].EnergyStore = lambda n: None
    sys.modules["scipy.signal"].lfilter = None
import numpy as np, pygame
os.chdir(REPO)
import loudness_meter as lm

with wave.open(os.path.join(REPO, "test_audio", "phase_test.wav")) as w:
    x = np.frombuffer(w.readframes(w.getnframes()), "<i2").reshape(-1, 2).astype(np.float32) / 32768

# (段名, corr 範圍, NOTCH 範圍 or None＝應為 "---", BAL)；corr 理論值見 make_phase_test.py
EXPECT = [
    ("1 同相",     (0.98, 1.01),   (2.5, 3.1),   "0"),
    ("2 極性接反", (-1.01, -0.98), (-30, -30),   "0"),
    ("3 只有左",   (0.0, 0.0),     None,         "L"),
    ("4 只有右",   (0.0, 0.0),     None,         "R"),
    ("5 不相關",   (-0.1, 0.1),    (-3.0, 0.5),  "0"),
    ("6 立體聲",   (0.7, 0.9),     (1.5, 3.0),   "0"),
    ("7 梳狀濾波", (0.0, 0.3),     (-30, -12),   "0"),
    ("8 正常擺位", (-0.1, 0.3),    (-6.0, -1.5), "L 10"),
]
out_dir = sys.argv[1] if len(sys.argv) > 1 else None
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    pygame.init(); W, H = 1280, 720
    screen = pygame.display.set_mode((W, H)); fonts = lm.make_fonts(W); cache = {}

SEG, STEP = 9 * 48000, lm.BLOCK_SIZE          # 每段 8 秒＋1 秒靜音
fail = 0
for k, (name, c_rng, n_rng, bal) in enumerate(EXPECT):
    seg = x[k * SEG:k * SEG + 8 * 48000]
    for i in range(0, len(seg) - STEP + 1, STEP):   # 一個 block 一幀（20 FPS）
        blk = seg[i:i + STEP]
        lm.buf_m_chunks.append(blk); lm.buf_s_chunks.append(blk)
        ph = lm.compute_phase()
        if out_dir and 5 * 48000 <= i < 6 * 48000:  # 出圖前逐幀畫 1 秒，殘影層才會把上一段淡掉（正式程式本來就每幀畫）
            lm.draw_phase_screen(screen, fonts, cache, W, H, ph, "gonio")
        if out_dir and i == 6 * 48000:              # 第 6 秒：殘影與平滑都穩定了
            for view in ("gonio", "band"):
                screen.fill((15, 15, 15))
                lm.draw_phase_screen(screen, fonts, cache, W, H, ph, view)
                pygame.image.save(screen, os.path.join(out_dir, f"phase_{k + 1}_{view}.png"))
        if i == 7 * 48000:                          # 第 7 秒檢查（段尾 20 ms 有淡出，避開）
            break
    n = ph["notch"]
    ok = (c_rng[0] <= ph["corr"] <= c_rng[1]
          and (n is None if n_rng is None else (n is not None and n_rng[0] <= n <= n_rng[1]))
          and ph["bal"] == bal)
    fail += not ok
    band = " ".join("  . " if b is None else f"{b:+.1f}" for b in ph["band"][7:22])   # 125 Hz～3.15 kHz
    print(f"{'OK ' if ok else 'NG '} {name:8} corr {ph['corr']:+.2f}  min3 {ph['min3']:+.2f}  "
          f"notch {'---' if n is None else f'{n:5.1f}'}"
          f"@{'-' if ph['notch_i'] is None else lm.OCTAVE_BANDS[ph['notch_i']][0]}  bal {ph['bal']:5}| {band}")
    for _ in range(20):                              # 段間靜音 1 秒
        z = np.zeros((STEP, 2), np.float32); lm.buf_m_chunks.append(z); lm.buf_s_chunks.append(z)
        lm.compute_phase()
assert fail == 0, f"{fail} 段不符預期"
print("全部符合預期")
