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
- meter bar 顯示當前電平
- 底部顯示 **L / R 個別聲道電平**，方便監看立體聲平衡

### SHORT TERM 3s（短期）
- 3 秒滑動窗口，反應近期平均音量
- meter bar 顯示當前電平

### THIS HOUR / SEGMENT（第三面板，僅顯示數字）

**THIS HOUR（上半）**
- EBU R128 兩段 gating 積分響度，整點自動歸零
- 靜音不影響數值（符合 EBU R128 規範，absolute gate 排除靜音）

**SEGMENT 3'（下半）**
- 3 分鐘滑動窗口的 gating 積分響度
- 設計用途：新聞播出時近似監看單則新聞帶響度（live read + VT 約 2'~3'）
- 無須手動標記段落頭尾，滑動窗口隨時反映最近 3 分鐘的整體響度

### 顏色規則（MOMENTARY / SHORT TERM）

| 顏色 | 範圍 |
|------|------|
| 綠色 | < −17 LUFS |
| 黃色 | −17 ~ −14 LUFS |
| 紅色 | > −14 LUFS |

白色橫線標示 **−23 LUFS** 目標值（EBU R128 廣播標準）。

---

## 每小時紀錄

程式會自動將每小時的 THIS HOUR 結果寫入：

```
~/loudness_meter/loudness_log.csv
```

格式：

```csv
date,hour,lufs
2026-04-08,14,-23.1
2026-04-08,15,-22.8
2026-04-08,16,---
```

- 整點觸發，寫入剛結束那個小時的最終積分響度
- ESC 退出時也會寫入目前這個小時（未完成）的數值
- `---` 代表該小時幾乎全為靜音（未通過 gating）

查看紀錄：

```bash
cat ~/loudness_meter/loudness_log.csv
```

---

## 技術架構

採用三執行緒設計，避免音訊 callback 過載：

```
音訊 callback（RT thread）
  → K-weighting + 寫入 deque（每 50ms，CPU 占用 < 1%）

compute_loop（daemon thread，10Hz）
  → 讀取 snapshot → pyloudnorm 計算 M/S → 自製 gating 計算 H 和 SEG

pygame main thread（20fps）
  → 讀取 latest dict → 渲染畫面
```

### 計算方法

| 指標 | 方法 | 窗口 |
|------|------|------|
| M | pyloudnorm `integrated_loudness()` | 400ms 滑動 |
| S | pyloudnorm `integrated_loudness()` | 3s 滑動 |
| H | 自製 EBU R128 two-stage gating | 整點起累積 |
| SEG | 同 H | 3 分鐘滑動 |
| L/R | K-weighted mean-square per channel | 400ms 滑動 |

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
- SEGMENT 為近似段落監看，無法精確對齊新聞帶頭尾
- True Peak 未實作（前端硬體設備已具備 peak 偵測功能）
