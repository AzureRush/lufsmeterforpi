"""離屏渲染 Histogram 畫面（SDL dummy driver，不碰 Wayland、不影響正在跑的響度計）。"""
import os
import sys

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
_scale = W / 1920
fs_value, fs_lr = max(80, round(250 * _scale)), max(50, round(140 * _scale))
fs_delta, fs_title, fs_scale = max(32, round(100 * _scale)), max(24, round(70 * _scale)), max(16, round(32 * _scale))
dseg = os.path.expanduser("~/loudness_meter/fonts/DSEG7Modern-Bold.ttf")
fonts = (
    pygame.font.SysFont("monospace", fs_title, bold=True),
    pygame.font.SysFont("monospace", fs_scale),
    pygame.font.Font(dseg, fs_value),
    pygame.font.Font(dseg, fs_lr),
    pygame.font.Font(dseg, fs_delta),
)
cache = {}


def fill(arr, values):
    arr[:] = 0
    for v in values:
        arr[lm._dist_bin(v)] += 1


rng = np.random.default_rng(3)
# 整點版：主體 -23、一段偏安靜的 -32、少量靜音、少量爆音（尾巴）
hour_vals = np.concatenate([rng.normal(-23, 2.0, 30000), rng.normal(-32, 3, 4000), np.full(150, -80.0), rng.normal(-10, 1.0, 60)])
# 滑動版：最近 3 分鐘整段偏安靜（-36 附近），像目前 PM5D 送的訊號
slide_vals = rng.normal(-36, 1.5, 1800)

out = sys.argv[1] if len(sys.argv) > 1 else "/tmp"

fill(lm._dist_hour, hour_vals)
fill(lm._dist_slide, slide_vals)
for mode in ("hour", "slide"):
    screen.fill((15, 15, 15))
    lm.draw_histogram_screen(screen, fonts, cache, W, H, mode, 11)
    pygame.image.save(screen, f"{out}/hist_{mode}.png")

# 剛清零：只有少量票 → LRA 應顯示 ---
lm._dist_hour[:] = 0
fill(lm._dist_hour, hour_vals[:50])
screen.fill((15, 15, 15))
lm.draw_histogram_screen(screen, fonts, cache, W, H, "hour", 11)
pygame.image.save(screen, f"{out}/hist_fresh.png")
print("rendered ok")
