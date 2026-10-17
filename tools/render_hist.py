"""離屏渲染 Spectrum 與 Histogram（時間軸）畫面存 PNG，用 SDL dummy driver，不碰 Wayland、不影響正在跑的響度計。

Pi 上跑：把 loudness_meter.py 放 /tmp/lmtest/，`python render_hist.py <輸出目錄> [spectrum]`。
加 spectrum 只渲染 Spectrum——改動 Spectrum 與 Histogram 共用的畫圖程式碼前後各跑一次，
兩張 spec.png 必須逐位元相同（資料是確定性的，不含亂數）。
"""
import datetime
import os
import sys
from unittest import mock

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
sys.path.insert(0, "/tmp/lmtest")

import numpy as np
import pygame

import loudness_meter as lm

pygame.init()
W, H = 1280, 720
screen = pygame.display.set_mode((W, H))

# 跟 main() 一樣的字型設定
fonts = lm.make_fonts(W)   # 字級與正式程式同一份（/tmp/lmtest 底下要有 fonts/ 才載得到 DSEG7）
cache = {}
out = sys.argv[1] if len(sys.argv) > 1 else "/tmp"

# ── Spectrum（確定性資料）──
left_w = W // 5
levels = [-60.0 + (i * 7) % 55 for i in range(len(lm.OCTAVE_BANDS))]
peaks = [v + 6 for v in levels]
screen.fill((15, 15, 15))
lm.draw_spectrum_panel(screen, fonts, cache, levels, left_w, W - left_w, 0, H, highlight_index=12, peak_levels=peaks)
pygame.image.save(screen, f"{out}/spec.png")
if len(sys.argv) > 2 and sys.argv[2] == "spectrum":
    print("rendered spectrum only")
    sys.exit(0)

# ── Histogram：合成節目（整點起 40 分鐘）：安靜 → 大聲 → 靜音空檔 → 中等，樣本間隔 1.14 秒 ──
NOW = datetime.datetime(2026, 9, 19, 12, 40, 0).timestamp()
t = NOW - 40 * 60 + 5 + np.arange(0, 40 * 60 - 5, 1.14)
minute = (t - (NOW - 40 * 60)) / 60
rng = np.random.default_rng(3)
s = np.select(
    [minute < 8, minute < 22, minute < 25],
    [-38 + rng.normal(0, 1.5, len(t)), -23 + rng.normal(0, 3, len(t)), np.full(len(t), -70.0)],
    default=-27 + rng.normal(0, 2, len(t)),
)
vals = {"M": -26.4, "S": -27.1}


def render(name, mode):
    screen.fill((15, 15, 15))
    with mock.patch("time.time", lambda: NOW):
        lm.draw_histogram_screen(screen, fonts, cache, W, H, mode, 12, vals)
    pygame.image.save(screen, f"{out}/{name}.png")


lm._tl_q.clear()
lm._tl_q.extend(zip(t.tolist(), s.tolist()))
render("hist_hour", "hour")
render("hist_slide", "slide")

# 剛整點：只有 30 筆樣本 → 圖是空的大半、LRA 顯示 ---
lm._tl_q.clear()
lm._tl_q.extend(zip(t[:30].tolist(), s[:30].tolist()))
lm._tl_memo["key"] = None
render("hist_fresh", "hour")
print("rendered ok")
