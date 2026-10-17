"""單一字元快取的自我檢查（需要真的 pygame，在 Pi 上跑）：

開機預熱（warm_glyph_cache）之後，把三個畫面用「所有可能出現的數值」各畫一遍，數字相關的快取張數必須一張都不增加
——執行中不再現場畫新數字，就不會再卡住音訊（2026-10 那 7 次溢位的成因）。標題裡帶小時數（THIS HOUR (N)）
本來就會隨整點多一張小字，不算在內。

用法：~/loudness_meter/venv/bin/python test_glyphs.py（loudness_meter.py 放同一個目錄）
"""
import os
import sys
import time

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pygame  # noqa: E402

import loudness_meter as lm  # noqa: E402

pygame.init()
W, H = 1280, 720
screen = pygame.display.set_mode((W, H))
fonts = lm.make_fonts(W)
cache = {}

t = time.monotonic()
lm.warm_glyph_cache(fonts, cache, W, H)
print(f"預熱：{len(cache)} 筆、{time.monotonic() - t:.2f} 秒")


def counted():
    return {k for k in cache if k[0] not in ("title", "hist_title")}


before = counted()
lufs = [float(v) for v in range(-75, 3)] + [-23.4, -22.6]               # 含 ≤ -70（---）與 > 0 的極端值
for v in lufs:
    vals = {"M": v, "S": v, "H": v, "SEG": v, "M_L": v, "M_R": v - 3}
    lm.draw_loudness_screen(screen, fonts, cache, W, H, vals, 12, v - 7, v + 12)   # 變化量 +7 / -12
    lm.draw_loudness_screen(screen, fonts, cache, W, H, vals, 12, v, v)            # 變化量 0
    for fi in range(lm.N_FREQ_STEPS):                                               # 所有頻段＋ALL
        peaks = [v] * len(lm.OCTAVE_BANDS)
        lm.draw_spectrum_screen(screen, fonts, cache, W, H, vals, peaks, peaks, fi)
for lra in [None] + [x / 10 for x in range(0, 400, 7)]:                            # LRA 0.0 ~ 39.9 與 ---
    lm._tl_memo["key"] = ("forced",)
    snap = lm._tl_build("slide", time.time())
    lm._tl_memo["key"], lm._tl_memo["val"] = ("slide", lm._tl_q[-1][0] if lm._tl_q else 0.0), snap[:4] + (lra,)
    lm.draw_histogram_screen(screen, fonts, cache, W, H, "slide", 12, {"M": -23.0, "S": -23.0})

grew = counted() - before
assert not grew, f"預熱後還多了 {len(grew)} 筆數字快取：{sorted(grew, key=str)[:5]}"
print(f"[OK] 預熱後畫遍所有數值：數字快取一筆都沒增加（{len(before)} 筆）")

lru = cache[("text_lru",)]
assert len(lru) <= lm._TEXT_LRU_MAX, f"整串快取超過上限：{len(lru)}"
print(f"[OK] 整串快取有上限：{len(lru)} / {lm._TEXT_LRU_MAX}")

# 沒命中時的成本：只用現成字元拼圖，要遠低於以前現場描邊＋縮放的數十毫秒
f_val = fonts[2]
sc = lm._fit_scale(f_val, lm._LUFS_REFS, 34, 406, 270)
ts = []
for v in range(-69, 0):
    lru.clear()
    t = time.perf_counter()
    lm.draw_glyph_text(screen, cache, f_val, str(v), (255, 255, 255), 34, sc, lm._WHITE_ONLY, 200, 200)
    ts.append((time.perf_counter() - t) * 1e3)
ts.sort()
print(f"[OK] 新數字（沒命中）拼圖耗時：中位數 {ts[len(ts) // 2]:.2f} ms、最大 {ts[-1]:.2f} ms")
assert ts[-1] < 10, "拼一張新數字不該超過 10 ms"

# 字元集外的字、調色盤外的顏色：不能讓繪圖迴圈崩潰
lm.draw_glyph_text(screen, cache, fonts[2], "-2?3", (1, 2, 3), 4, 0.5, lm._NUM_PALETTE, 100, 100)
print("[OK] 字元集外／調色盤外不會崩潰")

nbytes = sum(s.get_width() * s.get_height() * s.get_bytesize()
             for v in cache.values() for s in (v.values() if isinstance(v, dict) else (v,)))
print(f"快取總量 {nbytes / 1e6:.1f} MB（舊版連跑 6 天約 94 MB 且仍在長）")
print("\n全部通過")
