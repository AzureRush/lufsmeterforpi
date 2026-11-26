"""介紹短片用：把 test_audio/phase_test.wav 餵給響度計本身的 PHASE 畫圖程式，逐幀存 PNG（20 FPS，跟 Pi 上一樣）。
make_intro_video.py 讀這些 PNG 當 PHASE 兩段的畫面。

    python tools/render_phase_frames.py      # 需 numpy、pygame（跟 check_phase.py 同一個環境）；輸出 intro_src/phase/

畫面就是響度計真正畫出來的樣子，介面改了重跑一次即可。intro_src/ 在 .gitignore。
README 畫面四的 assets/phase_preview.gif 也是從這裡的 gonio 幀做的（5 fps、48 色，約 1.2 MB）：
    ffmpeg -framerate 20 -i intro_src/phase/gonio_%04d.png -vf "fps=5,split[a][b];[a]palettegen=max_colors=48:stats_mode=diff[p];[b][p]paletteuse=dither=none:diff_mode=rectangle" -loop 0 assets/phase_preview.gif
"""
import os, sys, types, wave
os.environ["SDL_VIDEODRIVER"] = "dummy"; os.environ["SDL_AUDIODRIVER"] = "dummy"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
try:
    import sounddevice, gpiozero, lufs_core, scipy.signal  # noqa: F401
except ImportError:   # ponytail: 筆電沒有這些模組，塞空殼只為了能 import（同 check_phase.py）
    for name in ("sounddevice", "gpiozero", "lufs_core", "scipy", "scipy.signal"):
        sys.modules[name] = types.ModuleType(name)
    sys.modules["gpiozero"].Button = object
    sys.modules["lufs_core"].EnergyStore = lambda n: None
    sys.modules["scipy.signal"].lfilter = None
import numpy as np, pygame
os.chdir(REPO)
import loudness_meter as lm

SR, STEP = 48000, lm.BLOCK_SIZE                  # 一個 block＝一幀＝50 ms
SEG = 9 * SR                                     # 測試音每段 8 秒＋1 秒靜音（make_phase_test.py）
# 每段影片：(檔名前綴, 頁, 預熱秒數, [(測試音段號 1~8, 秒數, 另加的不相關雜訊 dB 或 None), ...])
# 秒數要跟 make_intro_video.py 的字卡時間對齊
# 同相／極性接反是完全相關的測試音，goniometer 只畫出 1 px 的直線，影片裡幾乎看不到 → 加 −20 dB 各聲道獨立的雜訊，
# 變成看得見的細長橢圓（CORR 約 ±0.99，也比較像真實麥克風）；其他段本來就有雜訊，不加
CLIPS = [
    ("gonio", "gonio", 2.0, [(1, 4.0, -20), (2, 4.5, -20), (8, 6.5, None)]),   # 同相 → 極性接反 → 正常擺位，共 15 秒
    ("band",  "band",  2.0, [(7, 7.0, None), (8, 7.0, None)]),                 # 梳狀濾波 → 正常擺位，共 14 秒
]
rng = np.random.default_rng(0)


def piece(k, sec, noise_db):
    a = x[(k - 1) * SEG + SR // 2:(k - 1) * SEG + SR // 2 + int(sec * SR)]   # 段頭 0.5 秒有淡入，跳過
    if noise_db is None:
        return a
    rms = np.sqrt(np.mean(a ** 2, axis=0))
    return (a + rng.standard_normal(a.shape) * rms * 10 ** (noise_db / 20)).astype(np.float32)

with wave.open(os.path.join(REPO, "test_audio", "phase_test.wav")) as w:
    x = np.frombuffer(w.readframes(w.getnframes()), "<i2").reshape(-1, 2).astype(np.float32) / 32768

T = [0.0]
lm.time.monotonic = lambda: T[0]                 # ponytail: 假時鐘，MIN (3")／左欄每秒 4 次都照影片時間走
pygame.init(); W, H = 1280, 720
screen = pygame.display.set_mode((W, H)); fonts = lm.make_fonts(W); cache = {}
out = os.path.join(REPO, "intro_src", "phase"); os.makedirs(out, exist_ok=True)

for prefix, view, warm, parts in CLIPS:
    for f in os.listdir(out):
        if f.startswith(prefix + "_"):
            os.remove(os.path.join(out, f))
    # 每段從頭開始：清掉上一段留下的緩衝、殘影、MIN、平滑
    lm.buf_m_chunks.clear(); lm.buf_s_chunks.clear(); lm._phase_min_q.clear()
    lm._phase_num.update(t=-1.0, v=None); lm._phase_band_s[:] = 0; lm._gon["surf"] = None
    audio = np.concatenate([piece(*p) for p in [(parts[0][0], warm, parts[0][2])] + parts])
    n = 0
    for i in range(0, len(audio) - STEP + 1, STEP):
        T[0] += STEP / SR
        blk = audio[i:i + STEP]
        lm.buf_m_chunks.append(blk); lm.buf_s_chunks.append(blk)
        screen.fill((15, 15, 15))                 # 同主迴圈
        lm.draw_phase_screen(screen, fonts, cache, W, H, lm.compute_phase(view == "band"), view)
        if i >= warm * SR:
            pygame.image.save(screen, os.path.join(out, f"{prefix}_{n:04d}.png"))
            n += 1
    print(prefix, n, "frames")
