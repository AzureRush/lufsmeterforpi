# EBU R128 即時響度計

[![hackmd-github-sync-badge](https://hackmd.io/kbFwiQToSwiKuyRt7Xs71w/badge)](https://hackmd.io/kbFwiQToSwiKuyRt7Xs71w)

用小型系統架構提供持續的響度(Loudness)監測方案，提供只有 Mixer 系統內建 Peak Meter 環境下以外的儀表，讓聲音控制更有參考性。

本系統是在 Raspberry Pi 4B 上，以 python 為架構運行的即時 LUFS 監看工具，符合 EBU R128 / ITU-R BS.1770-4 廣播標準。

---

## 硬體需求

| 元件 | 規格 |
|------|------|
| 主機 | Raspberry Pi 4B |
| 作業系統 | Raspberry Pi OS Trixie（Debian 13）64-bit |
| 音訊介面 | Focusrite Scarlett 2i2 3rd Gen（USB Type-C → Type-A） |
| 音源輸入 | Yamaha PM5D Monitor Out L / R（XLR 類比）→ Scarlett 2i2 |
| 影像輸出 | Micro HDMI(Pi Out) → (optional)Datavideo DAC-9P（HDMI→SDI）→ (optional)KROMA LM6505 SDI 1 |
| 解析度 | (optional)1080i 60Hz（legacy firmware 模式） 此為配合副控顯示系統 |

除了 Raspberry Pi 以外的設備應都可自行更換並適應。

---

## 訊號串接

![訊號串接流程](https://raw.githubusercontent.com/AzureRush/lufsmeterforpi/refs/heads/main/assets/signal_chain.drawio.png)

---

## 安裝

```bash
pip3 install sounddevice pygame scipy numpy
```

---

## 執行

```bash
DISPLAY=:0 python3 ~/loudness_meter/loudness_meter.py
```

按 **ESC** 退出。

程式啟動時會自動偵測 Focusrite Scarlett，無需手動設定 device index。
若未偵測到 Scarlett，會列出所有可用輸入裝置並提示選擇：

```
Scarlett not found. Available input devices:
  1. USB Audio Device: Audio (hw:2,0)  (ch: 2)
  2. Built-in Audio: Audio (hw:0,0)   (ch: 2)

Select device [1-2]: _
```

輸入對應編號即可繼續執行，支援任何 ALSA 可識別的 USB 音訊介面。

---

## 顯示介面

![Demo](https://raw.githubusercontent.com/AzureRush/lufsmeterforpi/refs/heads/main/assets/loudness_meter_preview.gif)

畫面分為三個垂直面板：

### MOMENTARY（瞬時）

* 400ms 滑動窗口，即時反應當下響度
* meter 色塊 顯示當前響度
* 標準線以下區域顯示 **L / R 個別聲道響度值**（格式：`L -26` / `R -26`）

### SHORT TERM 3"（短期）

* 3 秒滑動窗口，反應近期平均響度
* meter 色塊 顯示近期平均響度
* 標準線以下區域顯示 **20 格歷史色塊**：每 3 秒推入一筆，右對齊 FIFO（先進先出），色彩對應當時的 3 秒 K-weighted 均方響度（無 gating）；高度線性對應 LUFS 值（0 LUFS 時滿格、頂至標準線，−41 LUFS 時為空）

### THIS HOUR / SEGMENT（第三面板，僅顯示數字）

**THIS HOUR（上半）**

* EBU R128 兩段 gating 積分響度，整點自動歸零
* 靜音不影響數值（符合 EBU R128 規範，absolute gate 排除靜音）
* 標題括號內顯示當前小時（如 `THIS HOUR (9)`），整點自動更新
* 左下角顯示 **delta 指標**：與上一個完整小時的差值（紅色 = 比上一個小時還高，綠色 = 比上一個小時還低）

**SEGMENT 3'（下半）**

* 3 分鐘滑動窗口的 gating 積分響度
* 設計用途：新聞播出時近似監看單則新聞帶響度（稿頭 + 新聞帶 約 2'~3'）
* 無須手動標記段落頭尾，滑動窗口隨時反映最近 3 分鐘的整體響度
* 左下角顯示 **delta 指標**：與上一個 3 分鐘狀態快照的差值（紅色 = 比上一個 3 分鐘還高，綠色 = 比上一個 3 分鐘還低）
* 底部顯示 **20 格歷史色塊**：每 3 分鐘推入一筆，右對齊 FIFO（先進先出），色彩對應當時的 3 分鐘 gating 積分響度

### 顏色規則（MOMENTARY / SHORT TERM）

| 顏色 | 範圍 |
|------|------|
| 藍色 | < −30 LUFS |
| 綠色 | −30 ~ −23 LUFS |
| 黃色 | −23 ~ −10 LUFS |
| 紅色 | > −10 LUFS |

白色橫線標示 **−23 LUFS** 目標值（EBU R128 廣播標準）。

---

## 自動輸出記錄檔

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

* 整點觸發，寫入剛結束那個小時的最終積分響度
* ESC 退出時也會寫入目前這個小時（未完成）的數值
* `---` 代表該小時幾乎全為靜音（未通過 gating）

查看紀錄：

```bash
cat ~/loudness_meter/loudness_log.csv
```

---

## 技術架構

採用三執行緒設計，避免音訊 callback 過載：

```
音訊 callback（RT thread）
  → H / SEG：有狀態 K-weighting，energy 直接寫入 deque
              （濾波器狀態須跨 block 連續，只能在此執行）
  → M / S  ：raw audio chunk 寫入 deque，交由 compute_loop 處理

compute_loop（daemon thread，10Hz）
  → M / S  ：對 raw audio 做無狀態 K-weighting → mean-square → LUFS
  → H / SEG：對預算 energy 套用自製 EBU R128 two-stage gating

pygame main thread（20fps）
  → 讀取 latest dict → 渲染畫面
```

### 計算方法

| 指標 | 方法 | 窗口 |
|------|------|------|
| M | K-weighted mean-square，無 gating（符合 R128） | 400ms 滑動 |
| S | K-weighted mean-square，無 gating（符合 R128） | 3s 滑動 |
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
SAMPLE_RATE   = 48000
BLOCK_SIZE    = 2400   # 50ms per block
MOMENTARY_WIN = 0.4    # 瞬時窗口（秒）
SHORTTERM_WIN = 3.0    # 短期窗口（秒）
TARGET_LUFS   = -23.0  # 目標線位置（LUFS）
```

Scarlett 裝置 index 由程式自動偵測，不需手動設定。

---

## 已知限制

* 三個 panel 的 title 文字在 266px 寬度下會溢出，目前以 clip 截斷
* SEGMENT 為近似段落監看，無法精確對齊新聞帶頭尾
* KROMA LM6505 有點毛病，單一 Monitor 雖有 SDI1/2 但若頻率無對齊就會出現握手問題，若發生無法握手則拔掉無法對齊的訊源並重開 KROMA LM6505。

---

## 參考(References)

* https://github.com/csteinmetz1/pyloudnorm
* [pyloudnorm: A simple yet flexible loudness meter in Python](https://csteinmetz1.github.io/pyloudnorm-eval/paper/pyloudnorm_preprint.pdf)
* [ITU-R Algorithms to measure audio programme loudness and true-peak audio level](http://magnetic.beep.pl/Loudness/2016/R-REC-BS.1770-4-201510.pdf)
* [DSEG Font](https://github.com/keshikan/DSEG)
