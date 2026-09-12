# Implementation Notes

紀錄 spec 沒有明確寫出的決策、trade-off、以及排錯過程中發現但文件沒提到的事。完整的分階段設計決策見 `C:\Users\User\.claude\plans\gpi-out-imperative-phoenix.md`（Plan Mode 產出，含 GPIO Spectrum Analyzer 功能的完整規劃與逐階段驗證紀錄）。這份檔案只記錄該計劃執行過程中，實際上機才發現、需要額外說明的部分。

## 2026-09-12（第二輪）：H / SEG 降頻至 1Hz，消除每小時週期性掉音訊

**發現經過**：修掉上一節的 log 風暴後，浮現出一條原本被掩蓋的曲線。因為 log 沒有時間戳，是靠每行的 `in Xs` 間隔累加還原出絕對時間軸（腳本留在 `/tmp/timeline.sh`），才看出**每小時固定重演同一個週期**：

| 整點後 | 12 點 | 13 點 | 14 點 | 15 點 |
|---|---|---|---|---|
| ～+27 分 | 靜默 | 靜默 | 靜默 | 靜默 |
| +28 分 | 開始 | 開始 | 開始 | 開始 |
| +59 分 | 4.85/s | 4.70/s | 4.77/s | 4.72/s |

四小時、四次重演，峰值誤差 <3%。整點前音訊 block 損失率達 **24%**——也就是 `THIS HOUR` 這個數字，**每小時有一半時間建立在缺損資料上**，且從功能上線以來一直如此。

**根因**：`compute_loop` 原本把四個指標綁在同一個 10Hz 迴圈。M / S 確實需要 10Hz，但 H / SEG 是搭便車的：
- `compute_h()` 是 O(n) 全量重算，`hour_block_energies` 從整點的 0 筆線性長到 72000 筆
- 其中 `np.array(python_list)` 這步**持有 GIL**，接近整點時單次要數十毫秒
- 10Hz 呼叫 = 長時間霸佔 GIL，音訊 callback 只能排隊 → input overflow
- 而那 9 次額外運算的結果其實無意義：整點後 30 分鐘時每次只多 2 筆（佔 0.005%），對 LUFS 的影響遠小於 0.01dB，而畫面只顯示到整數位

**修法**：H / SEG 降到 1Hz（`tick % 10`），連同 `list(deque)` 的 O(n) 複製一起移進去。M / S 維持 10Hz 不受影響。

**驗證**（17:00-18:00 完整小時，Spectrum 畫面，與修正前 12:00-13:00 同條件）：

| 整點後 | 修正前 | 修正後 |
|---|---|---|
| +44 分 | 0.39/s | 0.10/s |
| +53 分 | 0.71/s | 0.54/s |
| +56 分 | **2.83/s** | 0.51/s |
| +59 分 | **4.85/s** | **0.64/s** |

重點不只是峰值降 7.6 倍，而是**指數爆炸段完全消失**，只剩線性微升。資料損失 24% → 3.2%，瞬時 CPU 68% → 50%。

**踩到的坑（比較基準）**：中途曾誤判「修正無效」。原因是重啟會讓 `display_mode` 回到預設的 Loudness 畫面，我卻拿它去比修正前的 Spectrum 畫面數據，同時還只觀察到累積 33 分鐘（尚未進入失控區段）。**兩個變因同時不同，比較就沒有意義**——之後做效能對照一定要先確認畫面模式一致、且觀察期涵蓋完整的一小時週期。

**尚未處理的殘餘**：修正後仍有線性微升（0.10 → 0.64/s），源頭還是 `compute_h()` 單次成本隨累積成長。根本解是把 `hour_block_energies` 改成預先配置的 numpy 陣列（用索引寫入），讓 `compute_h` 直接切片、省掉每次 `np.array(list)` 這個 GIL-bound 轉換。但那要動到 RT 音訊路徑，複雜度與風險都高；目前 3.2% 的損失是否需要再降，留待判斷。

## 2026-09-12：`[AUDIO STATUS]` 寫 log 本身就是音訊溢位的元凶

**發現**：8/22 部署那支程式連續跑了 20 天 20 小時（0 例外、0 崩潰），但 `/tmp/lm.log` 長到 **66MB / 227 萬行，其中 99.99% 是 `[AUDIO STATUS] input overflow`**。實測當下發生率 **每秒 4.75 次**——音訊每秒 20 個 block，等於約 1/4 的 block 發生輸入溢位、資料被丟棄，響度讀值因此偏低、不能算準確。

**兩個互相加乘的問題**：

1. **RT callback 裡做同步 I/O**：`audio_callback` 原本是「只要 `status` 有值就 `print(..., flush=True)`」。`flush=True` 是阻塞式寫入，而它就發生在只有約 2ms 預算的音訊 callback 裡。於是形成正回饋：**溢位 → 寫 log → callback 更慢 → 更多溢位**。
2. **`/tmp` 在這台 Pi 上是 tmpfs（記憶體檔案系統，1.9G）**：那 66MB log 是實打實佔著 RAM。這正是程序記憶體從部署當天 5.5% 一路漲到 8.6% 的主因，不是程式漏記憶體。

**修法**：`status` 改成累計計數、每 60 秒最多印一行（附該期間累計次數），全域 `_status_n` / `_status_t` 各一個。RT callback 內只剩一次整數遞增與一次 `time.monotonic()`，不再有 I/O。

**實測效果**（同樣是無訊號、停在 Loudness 畫面的可比條件下）：

| 指標 | 修正前（跑 20 天） | 修正後（跑 1 分 49 秒） |
|---|---|---|
| input overflow | 4.75 次／秒 | **0 次**（log 完全空白） |
| CPU | 73.9% | **60.1%** |
| 溫度 | 58.4°C | **53.5°C** |
| 程序記憶體 | 8.6% | **5.5%** |

溢位歸零這點可以確定不是限流造成的假象——若仍有溢位，第一次發生後 60 秒內必定會印出一行，而觀察期已超過 60 秒仍是空的。

**尚待補驗**：上述數字是在「無音訊訊號 + Loudness 畫面（不跑 FFT）」的最輕負載下量的。有訊號且切到 Spectrum 畫面時是否仍維持 0 溢位，下次有訊號時要再量一次。

**順帶一提**：log 現在每分鐘最多一行，一天約 1440 行（~86KB），放 tmpfs 可接受，暫不做 logrotate。

## 2026-08-15：SIGUSR1/SIGUSR2 遠端觸發頻率步進（錄 README 用 GIF 的副產品）

錄 Spectrum 模式的 README 預覽 GIF 時，PM5D 端的 KEY10/KEY11 離操作者太遠、也很難精準對到截圖的時機。加了 `signal.signal(SIGUSR1, ...)`／`signal.signal(SIGUSR2, ...)`（[loudness_meter.py:756-759](loudness_meter.py:756)），直接呼叫跟按鍵回呼相同的 `_step_freq(1)`／`_step_freq(-1)`，可以用 `kill -USR1 <PID>`／`kill -USR2 <PID>` 遠端觸發，不用改 `_step_freq` 本身、也沒有新增執行緒。debug log 印出來的訊息仍會標示 `GPIO22`／`GPIO27`（`_step_freq` 內部固定用這兩個常數當標籤），透過訊號觸發時這個標籤其實是不準的、只是沿用同一份程式碼沒有另外處理——純粹是 log 可讀性上的小瑕疵，不影響實際行為。

這個功能目前保留在程式碼裡（不是錄完就刪），因為之後任何需要遠端展示/除錯頻率步進的場合都用得到，成本很低（4 行）。如果之後確定用不到，可以直接刪除。

**後續補充（2026-08-15）**：SIGUSR1/SIGUSR2 都拿去做頻率步進了，畫面切換（`_toggle_display_mode()`）另外借用 `SIGHUP`（`kill -HUP <PID>`）觸發。這支程式不是互動式終端機程式、不依賴收到真正 hangup 訊號做任何事，借用完全安全。三個訊號涵蓋了全部三顆 GPI 按鍵的等效遠端觸發（畫面切換／上一個/下一個頻率），之後展示或除錯都不用碰實體按鍵。

**踩到的坑（SSH 相關，跟先前 `pkill -f loudness_meter.py` 自我匹配是同一類問題）**：在同一段 SSH 遠端 script 裡用 `PID=$(pgrep -f loudness_meter.py)` 抓 PID，結果連線直接斷掉（exit 255、無輸出）——因為 `pgrep -f` 連執行這段 script 的 shell 自己都算進比對範圍（該 shell 的完整指令字串裡也含有 `loudness_meter.py`），抓到的 PID 混進了 shell 自己的 PID。`kill -USR1 $PID` 送到 shell 自己身上時，因為 shell 對 `SIGUSR1` 沒有自訂處理、預設動作是終止行程，就把整段 SSH session 砍斷了。**教訓**：在 SSH 遠端 script 裡若要對 `loudness_meter.py` 這個行程送訊號，一律用「先前一次獨立的 SSH 呼叫」查好的 PID 當常數帶進去，不要在同一段會被 `pgrep -f` 掃到自己的 script 裡現查。

## 2026-08-22：移除未使用的常數 `MOMENTARY_WIN` / `SHORTTERM_WIN`

整理 README 時發現這兩個常數**從專案早期起就沒有被任何程式碼引用過**，真正決定窗口長度的是 `M_BLOCKS = 8` / `S_BLOCKS = 60`（直接當 deque 的 `maxlen`）。README 的「設定」章節長期把它們列為可調參數，等於叫人改一個改了不會有效果的東西。已從程式碼刪除，`M_BLOCKS`/`S_BLOCKS` 補上換算註解（`8 × 50ms = 400ms`），README 設定章節改列實際生效的常數。

> ⚠️ **此版本尚未部署到 Pi**（2026-08-22 當下 Pi 已離線）。本機版本比 Pi 上執行中的版本新，差異為：SIGHUP 畫面切換觸發（已於離線前上機驗證過）＋本次常數刪除（僅 py_compile 驗證，**尚未實機執行驗證**）。下次連線時需 scp 補上並確認啟動正常。

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

（下面 Stage 4 把 `current_filter`／音訊輸出整個拔掉之後，這一格改成純粹「不選定特定頻段」的顯示狀態，`current_filter` 相關敘述已不成立，僅保留當時決策脈絡。）

## 2026-08-15：Stage 4 回退——移除音訊監聽輸出，改用 Scarlett Direct Monitor

**現象**：Stage 3 完成後實機操作，發現現場除了耳機還有一顆 main out 喇叭同時在響。Pi 雙工 stream 的來回延遲（USB round-trip＋50ms block）疊加房間裡即時的 main out 聲音，造成明顯 ~100ms 回音／山谷感，讓「戴耳機掃頻找刺耳頻率」這個原始目標大打折扣——延遲感本身比找頻率更干擾，判定不可用。

**決定**：耳機監聽改回 Scarlett 硬體 Direct Monitor（零延遲、不經過 Pi），Pi 完全不再輸出任何音訊，回到純視覺化診斷工具。找頻率的方式改成看 Spectrum 畫面＋搭配混音台自己的 EQ 掃頻用耳朵聽。

**保留**：長條圖／Peak hold／BAND dBFS／BAND FREQ／Y 軸、GPIO 三顆按鍵（含 wraparound、SIGUSR1/SIGUSR2 遠端觸發）——這些都跟音訊輸出無關，Stage 3 上機驗證時已證實運作良好。

**移除**：`design_bandpass_filter()`／`_new_filter_state()`／`current_filter`／`BANDPASS_Q`、雙工 `sd.Stream`（改回 `sd.InputStream`）、`audio_callback` 的輸出路由分支。**沒有留死代碼／註解占位**——這批程式碼完整保留在回退前的 git commit（`9299ece`，"feat: GPIO 觸發 Spectrum Analyzer 模式"）裡，之後如果找到更好的低延遲監聽方案，直接 `git show 9299ece:loudness_meter.py` 或 `git diff <回退後commit> 9299ece` 取回即可，不用重寫。

`SDL_AUDIODRIVER=dummy`（原本是為了避免 pygame.mixer 跟雙工 stream 搶 ALSA 裝置）保留下來——改回單工後理論上用不到，但零成本、不影響任何行為，留著當防禦性保險。

## 2026-08-15：Stage 5——BAND dBFS 改用 Peak hold 值、左側三格重新命名

**問題**：BAND dBFS（原名 F.F.dBFS）數字讀值原本用 EMA 平滑（`_smoothed_freq_dbfs()`），上機還是跳得太快看不清楚。Peak hold（`_spectrum_peaks`）已經解決了同樣的「跳太快看不清楚」問題，兩套機制功能重疊。

**改動**：刪掉 `_smoothed_freq_dbfs()` 跟 `_freq_dbfs_smoothed`／`_freq_dbfs_smoothed_idx` 兩個全域變數，數字讀值直接改成 `_spectrum_peaks[freq_index]`——`_spectrum_peaks` 本來就是對「每個頻段」即時追蹤（不是只追蹤目前選定那格），切換頻段時讀到的就是那個頻段當下真實的 peak hold 狀態，效果跟原本 EMA 版「切換頻段立刻重置」一致，不會有違和感。

**重新命名**：Stage 4 把濾波器整個拔掉後，"F." 字首（Filter 的縮寫）不再成立。改名：`MOMENTARY`（Spectrum 畫面左上角這個，Loudness 畫面自己的 MOMENTARY 面板不動）→ `ALL LUFS`（強調全頻率／寬頻，跟下面單一頻段的讀值做區隔）、`F.F.dBFS` → `BAND dBFS`、`F.FREQ` → `BAND FREQ`（"BAND" 字首彼此配對，語意上更準確反映「目前選定的那個頻段」）。三個新名稱都在 10 字元以內，符合這個窄欄位（W/5）的字型自動縮放限制。

## 2026-08-15：README.md 一段 GIF 連結被覆蓋成不相關內容

修改 README.md 的 Spectrum Analyzer 說明時，發現原本設定好的 GIF 連結（`assets/spectrum_analyzer_preview.gif`）已經被換成一個完全不相關的網址（`https://raw.githubusercontent.com/AzureRush/audiorookie/main/1786806821000r9bq2f.gif`，指向使用者另一個 repo、檔名像是某個貼圖工具自動產生的隨機檔名）。比對 `README.md.backup`（同目錄下一個未知來源的舊檔）發現同一段內容也是這個網址，研判是這個問題的線索，但確切原因不明。

本專案開發機在 Google 雲端硬碟同步資料夾裡（`G:\我的雲端硬碟\...`），這類非預期的檔案內容變化，過去在其他專案（x-broadcast）也發生過，懷疑跟雲端同步的版本衝突處理有關。**已修正回正確連結**，但這批被覆蓋的內容已經進了 `9299ece` 那次 commit——commit 本身沒有問題（內容仍是合法的 markdown），只是记錄了這個暫時性的錯誤狀態。若之後還發生類似情況，建議檢查 Google 雲端硬碟該檔案的版本歷史，確認是不是同步衝突造成的。

## 2026-08-15：SIGHUP 補上畫面切換的遠端觸發，Stage 4/5 上機驗證通過

SIGUSR1/SIGUSR2 已經用在頻率步進，畫面切換（`_toggle_display_mode()`）另外借用 `SIGHUP`（[loudness_meter.py:709](loudness_meter.py:709)，`kill -HUP <PID>`）——原因跟安全性見上面「SIGUSR1/SIGUSR2」那節的補充。部署後實測：`kill -HUP <PID>` 成功把畫面從 Loudness 切到 Spectrum（screenshot 確認），三顆 GPI 按鍵現在都有對應的遠端訊號觸發，之後展示/除錯不用再麻煩使用者碰實體按鍵。

Stage 4（移除音訊監聽輸出）＋ Stage 5（BAND dBFS 改用 peak hold、三格改名）上機驗證結果：
- `sd.InputStream` 開啟正常（無 PortAudioError），CPU 65-73%、溫度 55-56°C，跟基準同量級
- Spectrum 畫面截圖確認：ALL LUFS／BAND dBFS／BAND FREQ 三個新名稱正確顯示，Peak hold 白線正常
- 使用者實機戴耳機確認：Scarlett Direct Monitor 重新開啟後監聽零延遲、無回音感，Stage 4 回退達成原本目的
- Git：`9299ece`（Stage 3 完整版）→ `1f86263`（Stage 4/5 回退＋改名）→ `b696f52`（README 預覽 GIF 更新為回退後畫面，內容用 SIGUSR1 遠端步進＋`grim`連續截圖＋`ffmpeg`兩階段調色盤合成，做法已存成全域記憶 `feedback_remote_device_demo_gif.md`，供其他專案重複使用）
