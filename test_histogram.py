"""Histogram（時間軸）邏輯自我檢查：分欄、LRA（對 EBU Tech 3342 官方最低要求測試案例）、整點／滑動視窗。

直接 import 真的 loudness_meter（不是複製一份）——頂層那些硬體相關套件用 stub 擋掉，
這樣改了 loudness_meter.py 的 histogram 邏輯，這裡不用同步更新。
"""
import datetime
import os
import sys
from unittest import mock

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "lufs_core", "target", "release"))   # 筆電上用 cargo build 出來的 .pyd
for name in ("sounddevice", "pygame", "gpiozero"):
    sys.modules[name] = mock.MagicMock()
try:
    import scipy.signal  # noqa: F401
except ImportError:
    sys.modules["scipy"] = mock.MagicMock()
    sys.modules["scipy.signal"] = mock.MagicMock()

import loudness_meter as lm  # noqa: E402

# 1) tl_columns：欄位歸屬、欄內取最大值、無樣本的欄是 NaN、視窗外（前後）的樣本忽略
t = np.array([999.0, 1000.0, 1359.9, 1360.0, 4599.0, 4600.0])
s = np.array([-10.0, -30.0, -25.0, -40.0, -20.0, -15.0])
cols = lm.tl_columns(t, s, t0=1000.0, span=3600.0, n=10)      # 每欄 360 秒
assert cols[0] == -25.0, "1000 與 1359.9 在同一欄，應取最大值 -25"
assert cols[1] == -40.0, "1360 是第 1 欄的第一個樣本"
assert cols[9] == -20.0, "4599 在最後一欄"
assert np.isnan(cols[2:9]).all(), "沒有樣本的欄要是 NaN"
# 999（視窗前）與 4600（視窗後）都不能被算進任何一欄——-10 與 -15 都不該出現
assert -10.0 not in cols and -15.0 not in cols, "視窗外的樣本被算進去了"
print("[OK] tl_columns：欄位歸屬／取最大／空欄／視窗外")


# 2) LRA：EBU Tech 3342 V4 Table 1 的最低要求測試案例（各段 100 筆；段間的 LU 差就是預期的 LRA，容許 ±1）
def seg(*levels):
    return np.repeat(np.array(levels, dtype=float), 100)


for name, levels, want in [("#1 兩段差 10 LU", (-23, -33), 10), ("#2 兩段差 5 LU", (-20, -15), 5),
                           ("#3 兩段差 20 LU", (-40, -20), 20), ("#4 五段（含被相對閘門擋掉的兩段）", (-50, -35, -20, -35, -50), 15)]:
    got = lm.tl_lra(seg(*levels))
    print(f"    LRA {name}: {got:.2f}（預期 {want} ±1）")
    assert abs(got - want) <= 1, f"{name}：LRA {got} 不在 {want} ±1 內"
assert lm.tl_lra(np.full(500, -70.0)) is None, "全靜音（≤ -70 被絕對閾值擋掉）不該算 LRA"
assert lm.tl_lra(seg(-23, -33)[::4]) is None, "過閘後樣本 < 60 筆不該算 LRA"
print("[OK] LRA：Tech 3342 四個測試案例／全靜音／樣本不足")


# 3) 視窗選取：整點版 = 本整點起（整點一過自然清空）；滑動版 = 最近 10 分鐘
def at(h, m, sec=0):
    return datetime.datetime(2026, 9, 19, h, m, sec).timestamp()


lm._tl_q.clear()
lm._tl_q.extend((x, -25.0) for x in [at(11, 59, 59), at(12, 0), at(12, 37, 2), at(12, 37, 3), at(12, 39, 59)])
t0, span, t, s, _ = lm._tl_build("hour", at(12, 40))
assert (t0, span, len(t)) == (at(12, 0), 3600.0, 4), "整點版：11:59:59 是上個整點的，不該算"
# 滑動版：60 根 × 3 秒，最右一根是 12:40:00~12:40:03 這格 → 起點 = 12:40:00 - 59 × 3 = 12:37:03
t0, span, t, s, _ = lm._tl_build("slide", at(12, 40))
assert (t0, span, len(t)) == (at(12, 37, 3), 180.0, 2), "滑動版：12:37:02 已在 3 分鐘窗口外"


def col_of(sample_t, now):
    t0, span, *_ = lm._tl_build("slide", now)
    c = lm.tl_columns(np.array([sample_t]), np.array([-25.0]), t0, span, lm._TL_SLIDE_N)
    return int(np.flatnonzero(~np.isnan(c))[0])


# 對齊 3 秒：同一筆樣本在同一個 3 秒內（下一幀、再下一幀）不能換欄，跨過 3 秒邊界剛好往左移一欄——否則畫面會閃
x = at(12, 39, 59)
assert col_of(x, at(12, 40)) == col_of(x, at(12, 40) + 0.2) == col_of(x, at(12, 40) + 2.9) == 58
assert col_of(x, at(12, 40, 3)) == 57, "跨過 3 秒邊界要剛好往左一欄"
t0, span, t, s, lra = lm._tl_build("hour", at(13, 0, 1))
assert len(t) == 0 and lra is None, "整點一過，整點版要自然清空"
lm._tl_q.clear()
_, _, t, s, lra = lm._tl_build("hour", at(12, 40))
assert len(t) == 0 and lra is None, "沒有任何樣本時不能出錯"
print("[OK] 視窗選取：整點版／滑動版／整點清空／空序列")

print("\n全部通過")
