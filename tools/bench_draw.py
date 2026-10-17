"""Loudness 畫面逐元件繪圖成本（離屏，SDL dummy driver，不影響正在跑的響度計）。

Pi 上跑：把 loudness_meter.py 放 /tmp/lmtest/，`~/loudness_meter/venv/bin/python bench_draw.py`。
量的是穩態每幀成本（surf_cache 已暖），不含 display.flip()——dummy driver 沒有真的送畫面。
"""
import os
import sys
import time

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
sys.path.insert(0, "/tmp/lmtest")

import pygame  # noqa: E402

import loudness_meter as lm  # noqa: E402

pygame.init()
W, H = 1280, 720
screen = pygame.display.set_mode((W, H))
pw, half_h = W // 3, H // 2

fonts = lm.make_fonts(W)   # 字級與正式程式同一份（/tmp/lmtest 底下要有 fonts/ 才載得到 DSEG7）
cache = {}
vals = {"M": -26.0, "S": -27.0, "M_L": -29.0, "M_R": -30.0, "H": -25.0, "SEG": -26.0}

BAR_Y, PAD = 85, 6
BAR_H = H - BAR_Y - PAD
ty = BAR_Y + BAR_H - int(BAR_H * lm._y_ratio(-23.0, -60.0, 0.0))


def ms(fn, n=300):
    for _ in range(3):
        fn()                          # 暖身：surf_cache 填好
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    ts.sort()
    return ts[len(ts) // 2] * 1e3


def panel_m():
    lm.draw_panel(screen, fonts, cache, "MOMENTARY", vals["M"], 0, pw, H, lr_vals=(vals["M_L"], vals["M_R"]))


def panel_s():
    lm.draw_panel(screen, fonts, cache, 'SHORT TERM (3")', vals["S"], pw, pw, H)


def panel_hour():
    lm.draw_number_only_panel(screen, fonts, cache, "THIS HOUR (12)", vals["H"], 2 * pw, pw, 0, half_h,
                              delta=1, num_color=lm.lufs_to_color(vals["H"]))


def panel_seg():
    lm.draw_number_only_panel(screen, fonts, cache, "SEGMENT (3')", vals["SEG"], 2 * pw, pw, half_h, H - half_h,
                              delta=-1, delta_align='side')


def frame():
    screen.fill((15, 15, 15))
    panel_m()
    panel_s()
    panel_hour()
    panel_seg()
    for i in (1, 2):
        pygame.draw.line(screen, (55, 55, 55), (i * pw, 0), (i * pw, H), 1)
    pygame.draw.line(screen, (55, 55, 55), (2 * pw, half_h), (2 * pw + pw, half_h), 1)


def bars_only():                      # 兩格大跳動 BAR（底色 + 填色 + -23 線）
    for px in (0, pw):
        pygame.draw.rect(screen, (28, 28, 28), (px + PAD, BAR_Y, pw - PAD * 2, BAR_H))
        pygame.draw.rect(screen, (50, 200, 80), (px + PAD, ty, pw - PAD * 2, BAR_H - (ty - BAR_Y)))
        pygame.draw.line(screen, (255, 255, 255), (px + PAD, ty), (px + pw - PAD, ty), 4)


for name, fn in [
    ("整張畫面", frame),
    ("  screen.fill", lambda: screen.fill((15, 15, 15))),
    ("  MOMENTARY 格（含大 BAR）", panel_m),
    ("  SHORT TERM 格（含大 BAR）", panel_s),
    ("  THIS HOUR 格", panel_hour),
    ("  SEGMENT 格", panel_seg),
    ("  兩格大跳動 BAR（底+填+線）", bars_only),
]:
    print(f"{name:34} {ms(fn):7.2f} ms")
