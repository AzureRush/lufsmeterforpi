"""Histogram 邏輯自我檢查：LRA 對得上精確解、滑動版進出帳不漂移、邊緣併根、整點清零。

直接 import 真的 loudness_meter（不是複製一份）——頂層那些硬體相關套件用 stub 擋掉，
這樣改了 loudness_meter.py 的 histogram 邏輯，這裡不用同步更新。
"""
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

clock = [0.0]


def reset():
    lm._dist_hour[:] = 0
    lm._dist_slide[:] = 0
    lm._dist_slide_q.clear()
    lm._dist_hour_seen = lm.current_hour
    clock[0] = 0.0


def feed(values, dt=0.1):
    with mock.patch("time.monotonic", lambda: clock[0]):
        for v in values:
            lm._dist_feed(float(v))
            clock[0] += dt


def exact_lra(values):
    """不分桶、直接對原始值算的 Tech 3342（對照組）。"""
    v = np.asarray(values)
    v = v[v > -70.0]
    rel = lm._energy_to_lufs(np.mean(10 ** ((v + 0.691) / 10))) - 20.0
    v = v[v > rel]
    p10, p95 = np.percentile(v, 10), np.percentile(v, 95)
    return p95 - p10, p10, p95


rng = np.random.default_rng(7)
# 主體 -23、一段偏安靜的 -30、少量靜音（-80，低於絕對閾值）、少量爆音（-8）
vals = np.concatenate([rng.normal(-23, 2, 3500), rng.normal(-30, 3, 1200), np.full(200, -80.0), rng.normal(-8, 1, 100)])
rng.shuffle(vals)

# 1) 整點版 LRA / P10 / P95 對得上精確解（分桶誤差每個百分位 ≤ 0.05 LU，容許 0.15）
reset()
feed(vals)
got, want = lm.dist_lra(lm._dist_hour.copy()), exact_lra(vals)
print("LRA/P10/P95  histogram:", np.round(got, 2), " exact:", np.round(want, 2))
assert all(abs(g - w) < 0.15 for g, w in zip(got, want)), "LRA/P10/P95 跟精確解差太多"

# 2) 滑動版進出帳不漂移：窗口內的票數 == 「從頭重掃最近 180 秒」的結果
tail = vals[-int(lm._DIST_SLIDE_SEC / 0.1):]          # 餵了 5000 票、每票 0.1 秒，窗口內應剩最後 ~1800 票
recount = np.bincount([lm._dist_bin(v) for v in tail], minlength=lm._DIST_N)
assert abs(int(lm._dist_slide.sum()) - len(tail)) <= 1, f"窗口票數 {lm._dist_slide.sum()} != {len(tail)}"
assert int(lm._dist_slide.sum()) == len(lm._dist_slide_q), "計數與佇列長度不一致（退票漏了）"
assert np.abs(lm._dist_slide - recount).sum() <= 2, "進出帳跟重掃結果不一致（有漂移）"
print("[OK] 滑動版進出帳：窗口 %d 票，與重掃一致" % lm._dist_slide.sum())

# 3) 邊緣併根：低於閾值、超出兩側的票都要算進邊緣根，總票數守恆；-23 要落在第 27 根
reset()
feed([-80.0, -60.0, -50.4, -23.0, -23.4, -23.6, -9.6, 5.0])
bars = lm.dist_bars(lm._dist_hour.copy())
assert int(bars.sum()) == 8, "併根後總票數不守恆"
assert bars[0] == 3 and bars[-1] == 2, f"邊緣根 {bars[0]}/{bars[-1]} 不對（低於閾值/-60/-50.4 併最左；-9.6/+5 併最右）"
assert bars[27] == 2 and bars[26] == 1, "-23.0/-23.4 應在第 27 根、-23.6 應在第 26 根"
print("[OK] 邊緣併根與 -23 對位")

# 4) 整點清零：換小時後整點版只剩新票，滑動版不受影響
reset()
feed(vals[:100])
lm.current_hour = (lm.current_hour + 1) % 24
feed(vals[100:101])
assert int(lm._dist_hour.sum()) == 1 and int(lm._dist_slide.sum()) == 101, "整點清零行為不對"
print("[OK] 整點清零")

# 5) 票數不足 / 全靜音 → None（畫面顯示 ---）
reset()
feed(vals[:100])
assert lm.dist_lra(lm._dist_hour.copy()) is None, "不足 60 秒票數不該算 LRA"
reset()
feed(np.full(2000, -80.0))
assert lm.dist_lra(lm._dist_hour.copy()) is None, "全靜音不該算 LRA"
print("[OK] 票數不足 / 全靜音回 None")

print("\n全部通過")
