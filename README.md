# EBU R128 即時響度計

在 Raspberry Pi 4B 上運行的即時 LUFS 監看工具，符合 EBU R128 / ITU-R BS.1770-4 廣播標準。

---

## 硬體需求

| 元件 | 規格 |
|------|------|
| 主機 | Raspberry Pi 4B |
| 作業系統 | Raspberry Pi OS Bookworm 64-bit |
| 音訊介面 | Focusrite Scarlett 2i2 3rd Gen（USB） |
| 顯示器 | HDMI，800×480 |

**訊號鏈：** PC 音訊輸出 → Scarlett 2i2 類比輸入 → Pi → HDMI 顯示器

---

## 安裝

```bash
pip3 install sounddevice pyloudnorm pygame scipy numpy
```

確認 Scarlett 2i2 被辨識：

```bash
python3 -c "import sounddevice as sd; print(sd.query_devices())"
```

確認 device index 對應到 Scarlett（預設為 `DEVICE = 2`），若不同請修改 `loudness_meter.py` 頂部的設定。

---

## 執行

```bash
DISPLAY=:0 python3 ~/loudness_meter/loudness_meter.py
```

按 **ESC** 退出。

---

## 顯示介面

畫面分為三個垂直面板：

### MOMENTARY（瞬時）
- 400ms 滑動窗口，即時反應當下音量
- 底部顯示 **L / R 個別聲道電平**，方便監看立體聲平衡

### SHORT TERM 3s（短期）
- 3 秒滑動窗口，反應近期平均音量

### THIS HOUR（本小時）
- EBU R128 兩段 gating 積分響度，整點自動歸零
- 適合監看長時間節目的整體響度趨勢

### 顏色規則

| 顏色 | 範圍 |
|------|------|
| 綠色 | < −17 LUFS |
| 黃色 | −17 ~ −14 LUFS |
| 紅色 | > −14 LUFS |

白色橫線標示 **−23 LUFS** 目標值（EBU R128 廣播標準）。

---

## 技術架構

採用三執行緒設計，避免音訊 callback 過載：

```
音訊 callback（RT thread）
  → K-weighting + 寫入 deque（每 50ms，CPU 占用 < 1%）

compute_loop（daemon thread，10Hz）
  → 讀取 snapshot → pyloudnorm 計算 M/S → 自製 gating 計算 H

pygame main thread（20fps）
  → 讀取 latest dict → 渲染畫面
```

### 計算方法

- **M / S**：使用 `pyloudnorm` 對立體聲音訊執行 `integrated_loudness()`
- **H**：自製 EBU R128 兩段 gating——每個 50ms block 的 K-weighted 能量 `[E_L, E_R]` 存入 deque，整點清空重算
- **L / R 聲道電平**：從 400ms 窗口對各聲道獨立做 K-weighting（非 stateful），轉換為 dB 顯示，供平衡監看用（非嚴格 EBU R128 定義）

### 音量刻度（非線性）

```
−60 ~ −40 LUFS → 下方 20% 空間
−40 ~   0 LUFS → 上方 80% 空間
```

---

## 設定

`loudness_meter.py` 頂部可調整：

```python
DEVICE      = 2       # sounddevice 裝置 index
SAMPLE_RATE = 48000
BLOCK_SIZE  = 2400    # 50ms per block
TARGET_LUFS = -23.0   # 目標線位置
```

---

## 已知限制

- `compute_loop` 實際更新率約 4Hz（pyloudnorm 計算耗時），M 的 400ms 窗口在此速率下仍可接受
- 三個 panel 的 title 文字在 266px 寬度下會溢出，目前以 clip 截斷
- M/S 對短窗口使用 `integrated_loudness()` 是近似，非嚴格 R128 Momentary/Short-term 定義
- True Peak 未實作（前端硬體設備已具備 peak 偵測功能）
