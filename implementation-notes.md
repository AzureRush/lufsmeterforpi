# Implementation Notes

紀錄 spec 沒有明確寫出的決策、trade-off、以及排錯過程中發現但文件沒提到的事。完整的分階段設計決策見 `C:\Users\User\.claude\plans\gpi-out-imperative-phoenix.md`（Plan Mode 產出，含 GPIO Spectrum Analyzer 功能的完整規劃與逐階段驗證紀錄）。這份檔案只記錄該計劃執行過程中，實際上機才發現、需要額外說明的部分。

## 2026-08-15：SIGUSR1/SIGUSR2 遠端觸發頻率步進（錄 README 用 GIF 的副產品）

錄 Spectrum 模式的 README 預覽 GIF 時，PM5D 端的 KEY10/KEY11 離操作者太遠、也很難精準對到截圖的時機。加了 `signal.signal(SIGUSR1, ...)`／`signal.signal(SIGUSR2, ...)`（[loudness_meter.py:756-759](loudness_meter.py:756)），直接呼叫跟按鍵回呼相同的 `_step_freq(1)`／`_step_freq(-1)`，可以用 `kill -USR1 <PID>`／`kill -USR2 <PID>` 遠端觸發，不用改 `_step_freq` 本身、也沒有新增執行緒。debug log 印出來的訊息仍會標示 `GPIO22`／`GPIO27`（`_step_freq` 內部固定用這兩個常數當標籤），透過訊號觸發時這個標籤其實是不準的、只是沿用同一份程式碼沒有另外處理——純粹是 log 可讀性上的小瑕疵，不影響實際行為。

這個功能目前保留在程式碼裡（不是錄完就刪），因為之後任何需要遠端展示/除錯頻率步進的場合都用得到，成本很低（4 行）。如果之後確定用不到，可以直接刪除。

**踩到的坑（SSH 相關，跟先前 `pkill -f loudness_meter.py` 自我匹配是同一類問題）**：在同一段 SSH 遠端 script 裡用 `PID=$(pgrep -f loudness_meter.py)` 抓 PID，結果連線直接斷掉（exit 255、無輸出）——因為 `pgrep -f` 連執行這段 script 的 shell 自己都算進比對範圍（該 shell 的完整指令字串裡也含有 `loudness_meter.py`），抓到的 PID 混進了 shell 自己的 PID。`kill -USR1 $PID` 送到 shell 自己身上時，因為 shell 對 `SIGUSR1` 沒有自訂處理、預設動作是終止行程，就把整段 SSH session 砍斷了。**教訓**：在 SSH 遠端 script 裡若要對 `loudness_meter.py` 這個行程送訊號，一律用「先前一次獨立的 SSH 呼叫」查好的 PID 當常數帶進去，不要在同一段會被 `pgrep -f` 掃到自己的 script 裡現查。

## 2026-08-15：GPIO 上一個/下一個頻率腳位第二次對調

**發現過程**：階段 3（步進式帶通濾波器）測試時，KEY10（上一個頻率）一直有「按了沒反應」的現象，原本懷疑是排針接觸不良（2026-08-08 曾重插排線一次，當時看似有改善）。這次上機把 Pi 端 Pin15 的線頭拔下，用電表逐一對 D-sub 25 每支腳量測，發現這條線實際通到的是 PM5D Pin 6（GND），不是文件記載的 Pin 3（GPO5）——代表接線本身錯了，不是接觸不良。

使用者重新用電表逐一驗證四條線（線 A、線 B 各兩芯）之後，確認正確對應為：

| PM5D D-sub 25 | 功能 | Pi 實體排針 | BCM GPIO |
|---|---|---|---|
| Pin 2（GPO3, KEY9） | 畫面切換 | Pin 11 | GPIO17 |
| Pin 3（GPO5, KEY10） | 上一個頻率 | Pin 13 | GPIO27 |
| Pin 13（GPO12, KEY11） | 下一個頻率 | Pin 15 | GPIO22 |
| Pin 6 | GND | Pin 9 | — |

跟原本文件/程式碼記載的「上一個頻率＝GPIO22、下一個頻率＝GPIO27」剛好相反。

**改動**：`loudness_meter.py` 的 `GPIO_PIN_FREQ_PREV`／`GPIO_PIN_FREQ_NEXT` 兩個常數數值對調（17 不變）。純軟體修正，不需重新接線。

**為什麼會發生**：這是這個 GPI 接線案子第二次發生「兩條功能線對調」（第一次是 2026-08-08 的畫面切換／下一個頻率對調）。兩次都是同一類人為疏失——D-sub breakout board 上鎖線時，相鄰端子的線材外觀（顏色/線徑）太相似，肉眼確認容易出錯。**教訓**：往後若又新增 GPI 訊號線，應該在鎖上端子前就先用電表逐一驗證，而不是鎖完才測，可以省掉事後拆線重測的功夫。

## 2026-08-15（進行中）：Spectrum F.F.dBFS 讀值異常偏低，疑似聲道相位抵消

**現象**：PM5D 送 1kHz −20dBFS 測試音，濾波器選定 1kHz，F.F.dBFS 讀到 −50 左右（正確應該接近 −20，如同稍早階段 2 驗證過的 −20.5dBFS）。同時 MOMENTARY（LUFS，走完全不同的計算路徑）讀值正常，排除訊號源本身或接線總斷路的問題。

**根因假設**：`compute_spectrum_bands()`（[loudness_meter.py:548](loudness_meter.py:548)）目前是「左右聲道時域波形直接平均後才做 FFT」（`window.mean(axis=1)`）。若兩聲道相位沒有完全對齊（例如 XLR 其中一聲道熱冷線接反），平均時會互相抵消，抵消量取決於相位吻合程度，可以輕易造成 20~30dB 誤差。MOMENTARY 使用的 LUFS 計算路徑是「兩聲道各自算能量再相加」，天生不受相位影響，因此沒有這個問題——這也是判斷根因在 Spectrum 這條路徑、而非訊號源本身的關鍵依據。

**已排除的可能性**：稍早調整過 Spectrum 長條圖的 Y 軸非線性曲線（`_spectrum_y_ratio()`），該函式只負責畫面呈現（bar 高度、刻度線位置），跟 `compute_spectrum_bands()` 的數值計算完全獨立，不會互相污染。

**修法**：把 `compute_spectrum_bands()` 改成「兩聲道各自做 FFT 算功率、取平均」，取代「先在時域平均兩聲道再做 FFT」——功率平均不受相位影響，且兩聲道電平/相位正常時數值跟現在完全一致，不影響既有校準基準（`_DBFS_REF` 不用重新調）。已套用並部署（PID 159961），使用者上機用 PM5D 振盪器（−20dBFS）在 100Hz／1kHz／10kHz 三個頻率交叉驗證：

| 頻率 | LUFS 讀值 | F.F.dBFS 讀值 | K-weighting 理論預測（相對 1kHz） |
|---|---|---|---|
| 100Hz | −22 | −20 ✅ | 理論 −21.83（實測 −1.13dB rel. 1kHz 的 +0.70dB → −1.83dB） |
| 1kHz | −20 | −20 ✅ | 基準點 |
| 10kHz | −16 | −20 ✅ | 理論 −16.66（+4.04dB rel. 1kHz 的 +0.70dB → +3.34dB） |

F.F.dBFS 三個頻率都準確讀到 −20（跟 PM5D 送出的電平一致），確認修法有效。LUFS 隨頻率浮動（−22／−20／−16）不是 bug，是 ITU-R BS.1770-4 K-weighting 曲線本來就該有的頻率響應（低頻略衰減、高頻shelf 提升 ~4dB）——用專案既有的 K-weighting 係數實際算出的理論增益跟實測 LUFS 差距在 1dB 內（LUFS 顯示本身四捨五入到整數，誤差合理）。

## 2026-08-15：Stage 3 Checkpoint 3 驗證結果

- 音訊監聽：KEY10/KEY11 步進到對應頻率，耳機可清楚聽到該頻率被凸顯，且已排除 Scarlett Direct Monitor 造成的乾訊號疊加/delay 問題（使用者關閉硬體 Direct Monitor 後確認乾淨）
- GPIO 方向：`/tmp/lm.log` 確認 GPIO22＝next（delta=+1）、GPIO27＝prev（delta=−1），wraparound（29→0、0→29）行為正確
- `/tmp/lm.log` 出現 2 次 `[AUDIO STATUS] input overflow, output underflow`（約 10 分鐘密集按鍵測試期間）——次數少、非阻塞性，先記錄下來，之後若頻繁出現要回頭排查（懷疑跟密集連續切頻率時，按鈕回呼裡的濾波器係數重算與音訊 callback 搶時間有關）
- CPU/溫度（Spectrum 顯示＋濾波監聽同時開啟，實測 ~10 分鐘）：66.0% CPU、57.9°C，跟純 Loudness 畫面基準（~63-65%、~61-63°C）同量級，沒有明顯增量

## 2026-08-15：Peak hold ＋ 全頻率安全區

**Peak hold**：`_spectrum_peaks`（[loudness_meter.py:164](loudness_meter.py:164)）是跟頻段數量一樣長的浮點數 list，每幀原地更新，不開執行緒、不用鎖，跟現有 Spectrum 渲染在同一個 pygame 主迴圈內完成。回落速度用「每秒衰減 dB 數」設定，換算成每幀衰減量，沒有另外量測實際 frame 間隔時間——假設 FPS 穩定在 20，之前測過的 CPU/溫度數字都在正常範圍，沒有理由 frame time 會大幅飄移到需要精算 dt。

初版是「新高點立刻開始降」，上機測完使用者反應降得太快、想要碰到新高點後先頂住一段時間再降。改成 `_spectrum_peak_hold`（[loudness_meter.py:166](loudness_meter.py:166)）——跟 `_spectrum_peaks` 平行的一個「倒數計時」list，每個頻段各自記錄「還要再頂住幾幀」。新高點出現時該格計時器重設成 `_PEAK_HOLD_FRAMES`（1 秒份），計時器 >0 時完全不落下，歸零後才開始用 `_PEAK_DECAY_PER_FRAME` 往下降。最終參數（上機用眼睛調過、使用者確認沒問題）：**頂住 1 秒、回落速度 12dB/秒**（初版 24dB/秒太快，砍半）。

**全頻率安全區**：`BYPASS_INDEX`（[loudness_meter.py:122](loudness_meter.py:122)）是頻率清單之外多加的一格，`N_FREQ_STEPS` 因此變成 31。停在這格時 `current_filter` 設回 `None`，直接借用 `audio_callback` 原本「沒有濾波器就直通」的分支（[loudness_meter.py:257](loudness_meter.py:257)），沒有另外寫旁路邏輯。UI 上這格 F.FREQ 顯示 `ALL`、F.F.dBFS 顯示 `---`，長條圖不反白——`draw_spectrum_panel` 的 `highlight_index == i` 比對在 i 只跑 0~29 時自然不會命中 30，不用額外判斷式。因為插在清單最後一格、靠 wraparound 銜接，「20kHz 之後」跟「25Hz 之前」其實是同一格，不是兩個獨立安全區。
