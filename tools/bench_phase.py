"""PHASE 畫面（規劃中）繪圖成本評估：比較 Loudness（校準用，Pi 實測 8.1 ms）、Spectrum、PHASE（goniometer 各種畫法）每幀成本。

筆電：沒裝 sounddevice 等模組時自動塞空殼，`py -3.8 tools/bench_phase.py`（需 pygame、numpy）。
Pi：照 bench_draw.py 的方式放 /tmp/lmtest/ 執行，Pi 上的數字才是真的——筆電結果只能靠 Loudness 8.1 ms 換算。
"""
import os, sys, time, types
os.environ["SDL_VIDEODRIVER"] = "dummy"; os.environ["SDL_AUDIODRIVER"] = "dummy"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
try:
    import sounddevice, gpiozero, lufs_core, scipy.signal  # noqa: F401  Pi 上都有
except ImportError:   # ponytail: 筆電沒有這些模組，塞空殼只為了能 import 繪圖函式
    for name in ("sounddevice", "gpiozero", "lufs_core", "scipy", "scipy.signal"):
        sys.modules[name] = types.ModuleType(name)
    sys.modules["gpiozero"].Button = object
    sys.modules["lufs_core"].EnergyStore = lambda n: None
    sys.modules["scipy.signal"].lfilter = None
import numpy as np, pygame
os.chdir(REPO)
import loudness_meter as lm

pygame.init()
W, H = 1280, 720
screen = pygame.display.set_mode((W, H))
fonts = lm.make_fonts(W); cache = {}
vals = {"M": -26.0, "S": -27.0, "M_L": -29.0, "M_R": -30.0, "H": -25.0, "SEG": -26.0}
rng = np.random.default_rng(1)
for _ in range(60):
    lm.buf_s_chunks.append((rng.standard_normal((lm.BLOCK_SIZE, 2)) * 0.1).astype(np.float32))

def ms(fn, n=300):
    for _ in range(5): fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    ts.sort(); return ts[len(ts) // 2] * 1e3

def loud():
    screen.fill((15, 15, 15)); lm.draw_loudness_screen(screen, fonts, cache, W, H, vals, 12, -25.0, -26.0)

peaks = [-30.0] * len(lm.OCTAVE_BANDS)
def spec():
    screen.fill((15, 15, 15))
    b = lm.compute_spectrum_bands()
    lm.draw_spectrum_screen(screen, fonts, cache, W, H, vals, b, peaks, 5)

GS = 560
gon = pygame.Surface((GS, GS))
audio = np.concatenate(list(lm.buf_s_chunks))
HANN = np.hanning(4096)

def phase(npts, aa=False, fade=True):
    def f():
        screen.fill((15, 15, 15))
        third = H // 3
        for i, t in enumerate(("CORRELATION", 'MIN (3")', "BAL")):
            lm.draw_number_only_panel(screen, fonts, cache, t, None, 0, W // 5, i * third, third,
                                      val_str="+0.42", num_color=(255, 255, 255))
        if fade:
            gon.fill((200, 200, 200), special_flags=pygame.BLEND_RGB_MULT)   # 殘影：每幀乘 0.78
        else:
            gon.fill((0, 0, 0))
        seg = audio[-npts:]
        l, r = seg[:, 0], seg[:, 1]
        c, s = GS / 2, GS / 2 * 0.9 / (np.max(np.abs(seg)) * 1.4142 + 1e-9)
        pts = np.column_stack((c + (r - l) * 0.7071 * s, c - (l + r) * 0.7071 * s)).tolist()
        (pygame.draw.aalines if aa else pygame.draw.lines)(gon, (110, 220, 240), False, pts)
        screen.blit(gon, (264, 52))
        # 分頻 correlation：兩次 FFT ＋ 26 根長條 ＋ 底部 correlation 條
        A = np.fft.rfft(audio[-4096:, 0] * HANN); B = np.fft.rfft(audio[-4096:, 1] * HANN)
        _ = (A * B.conj()).real
        for i in range(26):
            pygame.draw.rect(screen, (60, 140, 90), (830 + i * 17, 200, 14, 130))
        pygame.draw.rect(screen, (28, 28, 28), (264, 640, 1008, 40))
        pygame.draw.rect(screen, (50, 200, 80), (768, 646, 200, 28))
    return f

rows = [("Loudness（校準，Pi 實測 8.1 ms）", loud), ("Spectrum（含 FFT）", spec)]
for n in (2400, 1200, 600):
    rows.append((f"PHASE 連線 {n} 點", phase(n)))
rows.append(("PHASE 反鋸齒連線 2400 點", phase(2400, aa=True)))
res = {}
for name, fn in rows:
    res[name] = ms(fn); print(f"{name:34} {res[name]:6.2f} ms")
k = 8.1 / res[rows[0][0]]
print(f"\nPi/筆電 倍率 ≈ {k:.1f}")
for name, _ in rows[1:]:
    print(f"{name:34} 推估 Pi {res[name] * k:5.1f} ms/幀")
# 只量 goniometer 本身（殘影＋連線＋blit），看它佔多少
for n in (2400, 1200):
    seg = audio[-n:]; l, r = seg[:, 0], seg[:, 1]
    def g(n=n):
        gon.fill((200, 200, 200), special_flags=pygame.BLEND_RGB_MULT)
        pts = np.column_stack((280 + (r - l) * 500, 280 - (l + r) * 500)).tolist()
        pygame.draw.lines(gon, (110, 220, 240), False, pts); screen.blit(gon, (264, 52))
    v = ms(g); print(f"  goniometer 本體 {n} 點 {v:5.2f} ms（推估 Pi {v * k:4.1f} ms）")
