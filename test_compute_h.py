"""compute_h() Python vs Rust 對照驗證——階段一唯一的驗收門檻。

不 import loudness_meter.py 本體：它頂層會 `from gpiozero import Button`，
在沒有 GPIO 硬體的筆電上會直接掛掉。以下兩個函式是從 loudness_meter.py
逐字複製過來的「Python 對照組」——loudness_meter.py 裡的 compute_h() /
_energy_to_lufs() 若日後改動，這裡也要同步更新，否則對照就失去意義。
"""
import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lufs_core", "target", "release"))
import lufs_core  # noqa: E402


# ── 以下逐字複製自 loudness_meter.py（_energy_to_lufs / compute_h） ──────────
def _energy_to_lufs(e):
    return -0.691 + 10.0 * np.log10(e)


def compute_h_python(block_energies, relative_gate_lu=10.0):
    if len(block_energies) < 2:
        return -70.0
    arr = np.array(block_energies)
    power = np.maximum(np.sum(arr, axis=1), 1e-12)
    lufs = _energy_to_lufs(power)
    m1 = lufs > -70.0
    if not np.any(m1):
        return -70.0
    rel = _energy_to_lufs(np.mean(power[m1])) - relative_gate_lu
    m2 = lufs > rel
    if not np.any(m2):
        return -70.0
    return _energy_to_lufs(np.mean(power[m2]))
# ──────────────────────────────────────────────────────────────────────────


def check(name, block_energies, gate=10.0, tol=1e-9):
    py = compute_h_python(block_energies, gate)
    rs = lufs_core.compute_h(block_energies, relative_gate_lu=gate)
    diff = abs(py - rs)
    status = "OK" if diff < tol else "FAIL"
    print(f"[{status}] {name}: python={py:.9f} rust={rs:.9f} diff={diff:.2e}")
    assert diff < tol, f"{name}: Python 與 Rust 版 compute_h 結果差異 {diff} 超過容許值 {tol}"


# 全靜音：都低於 -70 絕對閾值 → 應回 -70.0
silence = [[1e-13, 1e-13]] * 20
check("全靜音", silence)

# 單一爆音：其餘都安靜，中間夾一個顯著爆音——relative gate 應把爆音排除在外
quiet = [0.001, 0.001]
loud = [1.0, 1.0]
single_loud = [quiet] * 10 + [loud] + [quiet] * 10
check("單一爆音", single_loud)

# 正常分布：模擬節目訊號常見的動態範圍波動（約 -30~-20dB 能量量級）
rng = np.random.default_rng(42)
normal_energies = [[e, e] for e in rng.uniform(0.0005, 0.02, size=200)]
check("正常分布", normal_energies)

# 邊界附近：多數 block 落在同一能量量級，其中一個 block 的 LUFS 非常接近
# relative gate 門檻（但不用剛好卡在同一個浮點值——numpy 用 pairwise 加總、
# Rust 用線性加總，硬湊到位元級相等的邊界在兩種加總順序下可能各偏向一邊，
# 那只是加總順序造成的正常浮點誤差，不是這裡要驗的邏輯錯誤）。
base = [[0.01, 0.01]] * 30
near_gate_energy = 0.01 * (10 ** (-9.99 / 10))  # 比多數 block 低 9.99 LU，貼著 -10 LU 門檻但不重疊
edge_case = base + [[near_gate_energy, near_gate_energy]]
check("邊界附近", edge_case)

# ── EnergyStore：環形緩衝語意（同 deque(maxlen=cap)）＋ 算出的數字要跟對照組一致 ──
def check_store(name, items, cap, gate=10.0, tol=1e-9):
    store = lufs_core.EnergyStore(cap)
    for l, r in items:
        store.push(l, r)
    kept = items[-cap:]
    assert len(store) == len(kept), f"{name}: len {len(store)} != {len(kept)}"
    py = compute_h_python(kept, gate)
    rs = store.compute_h(gate)
    diff = abs(py - rs)
    print(f"[{'OK' if diff < tol else 'FAIL'}] EnergyStore {name}: python={py:.9f} rust={rs:.9f} diff={diff:.2e}")
    assert diff < tol, f"EnergyStore {name}: 差異 {diff} 超過容許值 {tol}"


pairs = [[float(e), float(e) * 1.1] for e in rng.uniform(0.0005, 0.02, size=500)]
check_store("未滿", pairs[:30], cap=50)
check_store("剛好滿", pairs[:50], cap=50)
check_store("繞圈（2.5 倍容量，擠掉最舊的）", pairs[:125], cap=50)
check_store("gate=8.0", pairs[:125], cap=50, gate=8.0)

store = lufs_core.EnergyStore(50)
for l, r in pairs[:20]:
    store.push(l, r)
store.clear()
assert len(store) == 0 and store.compute_h() == -70.0, "clear() 之後應該是空的、compute_h 回 -70.0"
store.push(*pairs[0])
assert store.compute_h() == -70.0, "只有 1 筆時應回 -70.0（同 len < 2 規則）"
store.push(*pairs[1])
assert abs(store.compute_h() - compute_h_python(pairs[:2])) < 1e-9, "clear() 後重新累積要能正常算"
print("[OK] EnergyStore clear() 後重用")

# 順便驗證這次查證出的官方值（10.0）跟現有 Python 版誤用的 8.0 差多少，
# 供之後決定要不要修正 Python 正式版時參考——這不是 pass/fail 門檻，只是列印。
py_8 = compute_h_python(normal_energies, 8.0)
py_10 = compute_h_python(normal_energies, 10.0)
print(f"\n[參考] 正常分布情境下，relative gate 8.0 LU vs 10.0 LU（官方值）："
      f" {py_8:.4f} vs {py_10:.4f}（差 {py_10 - py_8:+.4f} LU）")

print("\n全部通過")
