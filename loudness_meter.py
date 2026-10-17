import numpy as np
import sounddevice as sd
import pygame
import sys
import threading
import collections
import datetime
import math
import time
import os
import signal
from scipy.signal import lfilter
from gpiozero import Button
from lufs_core import EnergyStore   # Rust 擴充模組（lufs_core/），保管 block 能量並負責 H / SEG 的 gating 運算

# ─── Config ───────────────────────────────────────────────────────────────────
SAMPLE_RATE = 48000
CHANNELS    = 2

def select_input_device(devices, want="", interactive=False):
    """選音訊輸入裝置，回傳 sounddevice 的裝置編號：
    1. want（環境變數 LM_AUDIO_DEVICE，裝置名稱的一部分、不分大小寫）——換別台介面時用
    2. 沒設或找不到：Scarlett / Focusrite
    3. 都沒有：interactive（有鍵盤）才列清單詢問；否則（開機自動啟動）自動選第一個 ≥ 2 聲道的輸入裝置。
       以前這裡一律 input()，開機自動啟動時沒有 stdin → EOFError 被吃掉、無限迴圈卡死
    一個輸入裝置都沒有就結束程式（不要卡住）。"""
    inputs = [(i, d) for i, d in enumerate(devices) if d['max_input_channels'] > 0]
    if not inputs:
        print("ERROR: No input devices found.", flush=True)
        sys.exit(1)

    keys = ([want.lower()] if want else []) + ["scarlett", "focusrite"]
    for key in keys:
        for i, d in inputs:
            if key in d['name'].lower():
                print(f"Audio input: [{i}] {d['name']}" + (f"  (LM_AUDIO_DEVICE={want})" if key == want.lower() else ""), flush=True)
                return i
    if want:
        print(f"WARNING: LM_AUDIO_DEVICE={want!r} not found", flush=True)

    if not interactive:
        i, d = next(((i, d) for i, d in inputs if d['max_input_channels'] >= CHANNELS), inputs[0])
        print(f"Scarlett not found, no keyboard (autostart): using [{i}] {d['name']}"
              f"  — set LM_AUDIO_DEVICE to choose another", flush=True)
        return i

    print("Scarlett not found. Available input devices:")
    for n, (i, d) in enumerate(inputs, 1):
        print(f"  {n}. {d['name']}  (ch: {d['max_input_channels']})")
    while True:
        try:
            choice = int(input(f"Select device [1-{len(inputs)}]: "))
        except ValueError:
            choice = 0
        except EOFError:                 # Ctrl-D：使用者不選了
            sys.exit(1)
        if 1 <= choice <= len(inputs):
            i, d = inputs[choice - 1]
            print(f"Using: [{i}] {d['name']}")
            return i
        print(f"  Please enter a number between 1 and {len(inputs)}.")

BLOCK_SIZE  = 2400          # 50ms per block

TARGET_LUFS      = -23.0
BLUE_THRESHOLD   = -30.0
YELLOW_THRESHOLD = -23.0
RED_THRESHOLD    = -10.0
FPS              = 20   # Spectrum 畫面；peak hold 兩個常數由它推導，改它要一起看
# 各畫面幀率。Loudness 的 M / S 每秒只算 10 次，20 FPS 有一半的幀在重畫同樣的東西
# （2026-10-03 實測：主執行緒 39.1% 幾乎全是渲染）；Histogram 的資料每秒才變一次。
_SCREEN_FPS = {"loudness": 10, "spectrum": FPS, "histogram": 5}

# 窗口長度以「幾個 block」表示（BLOCK_SIZE = 50ms），直接決定 deque 的 maxlen
M_BLOCKS = 8    # 瞬時窗口 400ms
S_BLOCKS = 60   # 短期窗口 3s

# H / SEG 的相對閾值：ITU-R BS.1770-5 Annex 1 官方值，絕對閾值篩選後的平均再 −10 LU。
# （2026-09-19 前是 −8，那是 2011 年前的舊規範值；歷史與影響見 implementation-notes.md）
RELATIVE_GATE_LU = 10.0

# ─── GPIO（Spectrum Analyzer 觸發，訊源 Yamaha PM5D GPI OUT，UNLATCH 瞬時脈衝） ──
# 2026-08-15 用電表逐一重新驗證線路後再次修正：KEY10/KEY11（上一個/下一個頻率）
# 的線路跟原本文件記載的對調（KEY10 實際在 Pin13/GPIO27，KEY11 實際在 Pin15/GPIO22）。
GPIO_PIN_TOGGLE    = 17   # 實體 Pin 11，畫面切換（PM5D Pin2/GPO3/KEY9）
GPIO_PIN_FREQ_PREV = 27   # 實體 Pin 13，上一個頻率（PM5D Pin3/GPO5/KEY10）
GPIO_PIN_FREQ_NEXT = 22   # 實體 Pin 15，下一個頻率（PM5D Pin13/GPO12/KEY11）

# ─── Spectrum Analyzer（1/3 Octave 頻譜顯示，IEC 61260 標準頻率） ─────────────
SPECTRUM_FFT_SIZE  = 8192   # ~171ms @ 48kHz，頻率解析度 ~5.9Hz
# 範圍與曲線已依實測校準（2026-08-15）：-23 LUFS 節目訊號時各頻段實際落在 -40~-20dBFS。
# 比照 LUFS 錶那條非線性曲線的精神，但改成三段式：-70~-40／-20~0 兩端壓縮，
# -40~-20（訊號常駐區）給最多畫面空間，數字常駐在最好判讀的區域。
_SPECTRUM_DBFS_LO    = -70.0
_SPECTRUM_DBFS_SPLIT1 = -40.0
_SPECTRUM_DBFS_SPLIT2 = -20.0
_SPECTRUM_DBFS_HI    = 0.0
_SPECTRUM_FRAC_LOW   = 0.15   # -70..-40 佔畫面高度比例
_SPECTRUM_FRAC_MID   = 0.70   # -40..-20 佔畫面高度比例（訊號常駐區）
# 頂端 -20..0 自動是剩下的 1 - FRAC_LOW - FRAC_MID = 0.15
_SPECTRUM_DBFS_TICKS = [0, -20, -30, -40, -70]

_PEAK_HOLD_SEC        = 3.0    # peak 頂住不落下的時間（秒）。8/15 上機定 1 秒，2026-10-09 使用者改 3 秒
_PEAK_HOLD_FRAMES     = int(round(_PEAK_HOLD_SEC * FPS))
_PEAK_DECAY_PER_SEC   = 12.0   # 頂住時間過後的回落速度（dB/秒），上機用眼睛調
_PEAK_DECAY_PER_FRAME = _PEAK_DECAY_PER_SEC / FPS   # 假設穩定 FPS，不用另外量測 dt

# dBFS 校準：Parseval 能量守恆推導——頻段內用「加總」（不是平均）時，
# 校準基準是 N * sum(window**2) / 4，不是單一 bin 峰值公式，兩者相差 ~1.76dB，
# 已用合成正弦波（0dBFS/-6dBFS/-20dBFS，100Hz~16kHz）驗證誤差 <0.01dB。
_HANN_WINDOW        = np.hanning(SPECTRUM_FFT_SIZE)
_DBFS_REF           = SPECTRUM_FFT_SIZE * float(np.sum(_HANN_WINDOW ** 2)) / 4

_OCTAVE_BAND_CENTERS = [
    25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315,
    400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150,
    4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000,
]


def compute_octave_bands(sample_rate, fft_size):
    """1/3 Octave 頻段中心頻率與對應 FFT bin 範圍，程式啟動時算一次，不要每幀重算。"""
    bin_hz = sample_rate / fft_size
    bands = []
    for fc in _OCTAVE_BAND_CENTERS:
        f_lo = fc / (2 ** (1 / 6))
        f_hi = fc * (2 ** (1 / 6))
        bin_lo = max(1, int(round(f_lo / bin_hz)))
        bin_hi = max(bin_lo + 1, int(round(f_hi / bin_hz)))
        if bin_lo >= fft_size // 2:
            break
        bands.append((fc, bin_lo, bin_hi))
    return bands


OCTAVE_BANDS = compute_octave_bands(SAMPLE_RATE, SPECTRUM_FFT_SIZE)
BYPASS_INDEX = len(OCTAVE_BANDS)   # 頻率清單最後多一格「全頻率安全區」，不濾波、直通原始訊號
N_FREQ_STEPS = len(OCTAVE_BANDS) + 1


def _fmt_freq(hz):
    """頻率顯示格式：<1000Hz 顯示整數，>=1000Hz 除以 1000 顯示小數（不帶 k 字母，DSEG7 字型相容）。"""
    if hz < 1000:
        return f"{int(round(hz))}"
    v = hz / 1000
    return f"{v:.2f}".rstrip('0').rstrip('.')


# ─── K-weighting filter coefficients for 48kHz (ITU-R BS.1770-4) ─────────────
_PRE_B = np.array([1.53512485958697, -2.69169618940638,  1.19839281085285])
_PRE_A = np.array([1.0,              -1.69065929318241,  0.73248077421585])
_RLB_B = np.array([1.0,  -2.0,              1.0             ])
_RLB_A = np.array([1.0,  -1.99004745483398, 0.99007225036621])

# ─── Audio buffers ────────────────────────────────────────────────────────────
# Chunk-based: each entry is one 2400×2 block (not individual samples)
# Audio callback only appends; compute thread only reads snapshots.
buf_m_chunks = collections.deque(maxlen=M_BLOCKS)   # ~400ms
buf_s_chunks = collections.deque(maxlen=S_BLOCKS)   # ~3s
# K-weighted block 能量（[E_L, E_R]）由 Rust 保管，語意同 deque(maxlen=N)。資料放在 Python
# 這邊的話，每次算 compute_h 都要先整份轉成 Rust/numpy 資料，那步持有 GIL、成本隨累積量成長，
# 就是 2026-09 那批 input overflow 的根因；改成 callback 只 push、運算在釋放 GIL 後進行。
hour_store    = EnergyStore(72000)   # 1hr
segment_store = EnergyStore(3600)    # 3min sliding window

# K-weighting filter states — must stay in audio callback to stay continuous
_zi_pre = [np.zeros(2) for _ in range(CHANNELS)]
_zi_rlb = [np.zeros(2) for _ in range(CHANNELS)]

# audio callback 的 status（input overflow 等）限流用：累計次數 + 上次印出的時間
_status_n = 0
_status_t = time.monotonic()   # 每 60 秒最多印一行，附這段期間的累計次數

latest   = {"M": -70.0, "S": -70.0, "H": -70.0, "M_L": -70.0, "M_R": -70.0, "SEG": -70.0}
lock     = threading.Lock()

current_hour    = datetime.datetime.now().hour
reset_hour_flag = False
_running        = True

display_mode = "loudness"   # "loudness" / "spectrum" / "histogram" — GPIO 畫面切換鈕控制
freq_index   = 0            # 目前選定/反白的頻段索引，純顯示用（BAND dBFS／BAND FREQ）
hist_mode    = "hour"       # histogram 畫面的範圍："hour" 整點起 0~60 分 / "slide" 最近 10 分鐘（KEY10/11 切換）

# Peak hold：每個頻段一格，只在原地用索引更新內容，不重新賦值整個 list，不用 global。
_spectrum_peaks     = [_SPECTRUM_DBFS_LO] * len(OCTAVE_BANDS)
_spectrum_peak_hold = [0] * len(OCTAVE_BANDS)   # 每個頻段「還要再頂住幾幀」的倒數計時

prev_H   = None   # last completed hour's LUFS（整點 rollover 時更新）
prev_SEG = None   # last 3-min fixed segment's LUFS（每 3 分鐘快照）


# ─── Audio processing ─────────────────────────────────────────────────────────
def _energy_to_lufs(e):
    return -0.691 + 10.0 * np.log10(e)


def _fmt_lufs(v, prefix=""):
    return f"{prefix}{int(round(v))}" if v > -70 else f"{prefix}---"


def k_weight_and_energy(audio):
    """Apply stateful K-weighting; return [E_L, E_R] mean-square energies."""
    global _zi_pre, _zi_rlb
    energies = []
    for ch in range(CHANNELS):
        sig = audio[:, ch]
        sig, _zi_pre[ch] = lfilter(_PRE_B, _PRE_A, sig, zi=_zi_pre[ch])
        sig, _zi_rlb[ch] = lfilter(_RLB_B, _RLB_A, sig, zi=_zi_rlb[ch])
        energies.append(float(np.mean(sig ** 2)))
    return energies


# ─── LUFS Histogram（響度歷史圖：X = 時間、Y = 響度；S 每秒取樣一筆） ─────────────────
# (time.time(), S)。只在 S 窗口填滿時寫入——時間上的空缺本來就是圖上的空缺，不需要 NaN。
# compute_loop 的 1Hz 區塊實際約每 1.14 秒一次，一小時約 3200 筆，容量 4000 夠用。
# 滑動版：60 根 × 3 秒 = 3 分鐘。一根剛好是一個 Short Term 窗口長度、前後不重疊；
# 整個畫面對齊 SEGMENT (3')。
_TL_SLIDE_BIN = 3.0
_TL_SLIDE_N   = 60
_TL_SLIDE_MIN = 3
_TL_MIN_N     = 60                     # 過閘後樣本 < 60 筆（約 1 分鐘）不算 LRA，剛清零時的數字沒有意義
_tl_q         = collections.deque(maxlen=4000)
_tl_memo      = {"key": None, "val": None}   # 快照與 LRA 只在有新樣本時重算，每秒一次


def _tl_feed(s):
    with lock:
        _tl_q.append((time.time(), s))


def tl_lra(s):
    """EBU Tech 3342：絕對閾值 -70 → 相對閾值（過閾值樣本的平均能量再 -20 LU）→ P95 - P10。
    樣本不足回 None。相對閾值的基準是「這批 short-term 值自己的平均」，不是 H。"""
    s = s[s > -70.0]
    if len(s) == 0:
        return None
    g = s[s > _energy_to_lufs(np.mean(10.0 ** ((s + 0.691) / 10.0))) - 20.0]
    if len(g) < _TL_MIN_N:
        return None
    return float(np.percentile(g, 95) - np.percentile(g, 10))


def tl_columns(t, s, t0, span, n):
    """把 (時間, S) 依時間分進 n 欄，欄內取最大值；沒有樣本的欄是 NaN，視窗外的樣本忽略。"""
    cols = np.full(n, np.nan)
    idx  = np.floor((t - t0) / span * n).astype(int)   # 不能只用 astype(int)：它往 0 截，視窗起點之前的樣本會被算進第 0 欄
    ok   = (idx >= 0) & (idx < n)
    np.fmax.at(cols, idx[ok], s[ok])
    return cols


def _tl_build(mode, now):
    """整點版：視窗 = 本整點起 3600 秒（整點一過視窗自然移走，不需要清零邏輯）；滑動版：最近
    60 根 × 3 秒，起點對齊 3 秒整數倍、最右一根是正在累積的這 3 秒。不對齊的話（例如用 now - 180），
    視窗每幀平移，同一筆樣本會在相鄰兩根之間跳來跳去，畫面會閃。回傳 (t0, span, t, s, lra)。"""
    if mode == "hour":
        t0, span = datetime.datetime.fromtimestamp(now).replace(minute=0, second=0, microsecond=0).timestamp(), 3600.0
    else:
        t0   = (now // _TL_SLIDE_BIN - (_TL_SLIDE_N - 1)) * _TL_SLIDE_BIN
        span = _TL_SLIDE_N * _TL_SLIDE_BIN
    with lock:
        arr = np.array(_tl_q).reshape(-1, 2)   # n ≤ 4000，Pi 約 2.5ms
    keep = arr[:, 0] >= t0
    t, s = arr[keep, 0], arr[keep, 1]
    return t0, span, t, s, tl_lra(s)


def tl_snapshot(mode):
    with lock:
        last = _tl_q[-1][0] if _tl_q else 0.0
    if _tl_memo["key"] != (mode, last):
        _tl_memo["key"], _tl_memo["val"] = (mode, last), _tl_build(mode, time.time())
    return _tl_memo["val"]


def audio_callback(indata, frames, time_info, status):
    """Lightweight: K-weight + 2 deque appends + 2 store pushes. ~2ms per call 預算。
    單工 input-only stream——Spectrum 模式的耳機監聽改回 Scarlett 硬體 Direct
    Monitor（零延遲），Pi 完全不輸出音訊，純粹是視覺化診斷工具。"""
    global _zi_pre, _zi_rlb, reset_hour_flag, _status_n, _status_t

    # 溢位時每次都 print 會在 RT callback 裡做同步 I/O，反而加重溢位（而且 /tmp 是
    # tmpfs，log 直接吃 RAM）。改成計數、每分鐘最多印一行。
    if status:
        _status_n += 1
        now = time.monotonic()
        if now - _status_t >= 60:
            print(f"[AUDIO STATUS] {status} ×{_status_n} in {now - _status_t:.0f}s", flush=True)
            _status_n = 0
            _status_t = now

    audio = indata.copy()

    if reset_hour_flag:
        hour_store.clear()
        _zi_pre = [np.zeros(2) for _ in range(CHANNELS)]
        _zi_rlb = [np.zeros(2) for _ in range(CHANNELS)]
        reset_hour_flag = False

    energies = k_weight_and_energy(audio)   # ~1.6ms, stateful — must stay here
    buf_m_chunks.append(audio)              # deque.append is GIL-atomic
    buf_s_chunks.append(audio)
    hour_store.push(*energies)              # 兩個浮點數進 Rust，微秒級
    segment_store.push(*energies)


def compute_loop():
    """M / S 以 10Hz 更新；H / SEG 以 1Hz 更新。

    H / SEG 刻意降頻：compute_h() 是 O(n) 全量重算，而 hour_block_energies 會從
    整點的 0 筆線性長到 72000 筆。其中 np.array(list) 這步是持有 GIL 的，接近整點
    時單次要數十毫秒，10Hz 呼叫等於長時間霸佔 GIL，音訊 callback 只能排隊 → 造成
    input overflow。實測（2026-09-12，連續 4 小時）每小時固定重演同一條曲線：整點
    後靜默約 28 分鐘，之後指數惡化，整點前一分鐘達 4.8 次/秒（約 24% 音訊 block
    被丟棄），整點歸零後重來。H / SEG 都是長時間積分指標，1Hz 綽綽有餘。

    2026-09-19：block 能量改由 Rust 的 EnergyStore 保管，運算在釋放 GIL 之後進行，
    「整份資料轉換持有 GIL」這個根因已消失；1Hz 維持不動，沒有理由更快。"""
    tick = 0
    while _running:
        time.sleep(0.1)
        tick += 1

        # Snapshot buffer state — fast list copies, GIL-protected
        m_snap   = list(buf_m_chunks)
        s_snap   = list(buf_s_chunks)

        M = -70.0
        S = -70.0

        if len(m_snap) >= M_BLOCKS:
            m_audio = np.concatenate(m_snap)

            # K-weight 各聲道，energy 相加 → LUFS（無 gating，符合 R128 M 定義）
            e_total = 0.0
            for ch_idx, key in enumerate(("M_L", "M_R")):
                sig = lfilter(_PRE_B, _PRE_A, m_audio[:, ch_idx])
                sig = lfilter(_RLB_B, _RLB_A, sig)
                e   = float(np.mean(sig ** 2))
                e_total += e
                with lock:
                    latest[key] = _energy_to_lufs(e) if e > 1e-12 else -70.0
            M = _energy_to_lufs(e_total) if e_total > 1e-12 else -70.0

        if len(s_snap) >= S_BLOCKS:
            s_audio = np.concatenate(s_snap)
            e_total = 0.0
            for ch in range(CHANNELS):
                sig = lfilter(_PRE_B, _PRE_A, s_audio[:, ch])
                sig = lfilter(_RLB_B, _RLB_A, sig)
                e_total += float(np.mean(sig ** 2))
            S = _energy_to_lufs(e_total) if e_total > 1e-12 else -70.0

        with lock:
            latest["M"] = M
            latest["S"] = S

        # H / SEG 每秒一次即可（見 docstring）
        if tick % 10 == 0:
            H   = hour_store.compute_h(RELATIVE_GATE_LU)
            SEG = segment_store.compute_h(RELATIVE_GATE_LU)
            with lock:
                latest["H"]   = H
                latest["SEG"] = SEG
            if len(s_snap) >= S_BLOCKS:   # 窗口沒填滿 3 秒前的 S 是「沒資料」不是「靜音」，不進歷史圖
                _tl_feed(S)


# ─── GUI ──────────────────────────────────────────────────────────────────────
_DIRS16 = [(math.cos(i * math.pi / 8), math.sin(i * math.pi / 8)) for i in range(16)]


def _outline_layer(font, text, offset, outline_color=(0, 0, 0)):
    """黑色外框層：同一段文字往 16 個方向各蓋一次（雙半徑填補缺口），尺寸 = 文字 + 四周 offset。"""
    shadow = font.render(text, True, outline_color)
    surf   = pygame.Surface((shadow.get_width() + offset * 2, shadow.get_height() + offset * 2), pygame.SRCALPHA)
    for r in [offset, max(1, round(offset * 0.6))]:
        for dx, dy in _DIRS16:
            surf.blit(shadow, (round(dx * r) + offset, round(dy * r) + offset))
    return surf


def render_outlined(font, text, color, outline_color=(0, 0, 0), offset=2):
    """2 層渲染：黑色外框（16 方向均勻擴展）→ 主色填充。
    模擬 Photoshop 筆畫「外部」效果：主體大小不變，黑框只往外延伸。現在只用在軸文字（字串固定、張數有限）。"""
    surf = _outline_layer(font, text, offset, outline_color)
    surf.blit(font.render(text, True, color), (offset, offset))
    return surf


# ─── 大數字：單一字元快取（2026-10-09，C 方案） ─────────────────────────────────
# 舊做法以「整個字串」快取畫好的大數字：沒見過的字串（-24、L -33、LRA 4.7……）要現場畫——16 方向描邊加縮放，
# 數十毫秒、持有 GIL，音訊 callback 被擋住就 input overflow。實測 5 天 21 小時 7 次溢位，全部落在快取剛長的
# 10 分鐘區間（快取沒長的 799 個區間 0 次），快取也一路長到 688 張 / 94 MB。
# 改成每個字元（各字級各色）只畫一次、開機預熱（warm_glyph_cache），顯示時逐字拼：張數固定、執行中不再現場畫。
_GLYPH_CHARS   = "0123456789-. LRA"           # LUFS、L/R、LRA 小數、頻率（含 ALL）、變化量用到的全部字元
_NUM_PALETTE   = ((255, 255, 255), (220, 50, 50), (220, 200, 50), (50, 200, 80), (50, 130, 220))  # 白 ＋ lufs_to_color 四色
_WHITE_ONLY    = ((255, 255, 255),)
_DELTA_PALETTE = ((220, 50, 50), (110, 200, 110), (140, 140, 140))
# 每格的縮放比以「這格可能出現的最寬字串」決定、整格固定——字大小不會隨位數忽大忽小
_LUFS_REFS = ("-70", "---")
_LRA_REFS  = ("40.0", "---")
_FREQ_REFS = ("31.5", "12.5", "1.25", "800", "ALL")


def _glyph_set(cache, font, offset, sc, palette):
    """某字級（字型 × 描邊 × 縮放比）的整套字元：每字元一張描邊層、調色盤每色一張本體層，第一次用到時一次畫完。"""
    key = ('glyphs', id(font), offset, sc, palette)
    gs  = cache.get(key)
    if gs is None:
        gs = {}
        for ch in _GLYPH_CHARS:
            outline = _outline_layer(font, ch, offset)
            size    = (max(1, round(outline.get_width() * sc)), max(1, round(outline.get_height() * sc)))
            gs[ch, None] = outline if sc == 1 else pygame.transform.smoothscale(outline, size)
            for col in palette:
                body = pygame.Surface(outline.get_size(), pygame.SRCALPHA)
                body.blit(font.render(ch, True, col), (offset, offset))
                gs[ch, col] = body if sc == 1 else pygame.transform.smoothscale(body, size)
        cache[key] = gs
    return gs


def _fit_scale(font, refs, offset, box_w, box_h):
    w = max(font.size(s)[0] for s in refs) + offset * 2
    h = font.get_height() + offset * 2
    return min(box_w / w, box_h / h)


_TEXT_LRU_MAX = 48   # 拼好的整串字：畫面上同時只有十來個數字，48 張足夠讓穩定的讀值都命中


def _compose_text(gs, font, text, color, offset, sc, palette):
    """用已畫好的字元拼成一整張圖。字距照整串渲染的量測（font.size(前綴)），排版跟以前整串 render 一致。
    先畫完所有字的描邊、再畫所有本體——描邊往外擴，若一字一字各自畫完，下一字的描邊會蓋到上一字的本體。
    字元集外或顏色不在調色盤時略過／退回白色，不讓繪圖迴圈崩潰。"""
    w, h = font.size(text)
    surf = pygame.Surface((max(1, round((w + offset * 2) * sc)), max(1, round((h + offset * 2) * sc))), pygame.SRCALPHA)
    xs   = [round(font.size(text[:i])[0] * sc) for i in range(len(text))]
    for layer in (None, color):
        for x, ch in zip(xs, text):
            g = gs.get((ch, layer))
            if g is None and layer is not None:
                g = gs.get((ch, palette[0]))
            if g is not None:
                surf.blit(g, (x, 0))
    return surf


def draw_glyph_text(surface, cache, font, text, color, offset, sc, palette, cx, cy):
    """以 (cx, cy) 為中心畫一串字，回傳 (左, 上, 寬, 高)。兩層快取：
    1. 字元層（_glyph_set）：每個字元各色只畫一次、開機預熱、張數固定——昂貴的描邊＋縮放只發生在這裡
    2. 整串層（有上限的 LRU）：每幀只貼一張圖。逐字直接貼到畫面的話，每個字的描邊層四周都有一圈空白、
       一串要疊 6~8 張半透明圖，實測整張 Loudness 畫面從 7.7 → 16.1 ms/幀，所以多這一層。
       沒命中時只是用現成字元拼一張（幾次貼圖、約 1~2 ms），不是以前那種數十毫秒的現場描邊縮放。"""
    lru = cache.setdefault(('text_lru',), collections.OrderedDict())
    key = (id(font), offset, sc, palette, text, color)
    s   = lru.get(key)
    if s is None:
        s = _compose_text(_glyph_set(cache, font, offset, sc, palette), font, text, color, offset, sc, palette)
        lru[key] = s
        if len(lru) > _TEXT_LRU_MAX:
            lru.popitem(last=False)
    else:
        lru.move_to_end(key)
    w, h   = s.get_size()
    x0, y0 = cx - w // 2, cy - h // 2
    surface.blit(s, (x0, y0))
    return x0, y0, w, h


def lufs_to_color(val):
    if val >= RED_THRESHOLD:
        return (220, 50, 50)
    elif val >= YELLOW_THRESHOLD:
        return (220, 200, 50)
    elif val >= BLUE_THRESHOLD:
        return (50, 200, 80)
    else:
        return (50, 130, 220)


_SCALE_MARKS = [0, -10, -20, -23, -30, -40, -50, -60]
_SPLIT_DB    = -40.0
_SPLIT_FRAC  =  0.20


def _y_ratio(val, lo=-60.0, hi=0.0):
    """Non-linear map: -60..-40 → bottom 20%, -40..0 → top 80%. LUFS 錶專用（gating 特性）。"""
    val = max(lo, min(hi, val))
    if val <= _SPLIT_DB:
        return _SPLIT_FRAC * (val - lo) / (_SPLIT_DB - lo)
    else:
        return _SPLIT_FRAC + (1.0 - _SPLIT_FRAC) * (val - _SPLIT_DB) / (hi - _SPLIT_DB)


def _spectrum_y_ratio(val):
    """Spectrum 長條圖專用三段式曲線：-70~-40／-20~0 壓縮，-40~-20（訊號常駐區）給最多空間。
    刻度線用同一個函式算位置，所以畫面上的間距會跟著這條曲線走，不是等距。"""
    val = max(_SPECTRUM_DBFS_LO, min(_SPECTRUM_DBFS_HI, val))
    if val <= _SPECTRUM_DBFS_SPLIT1:
        return _SPECTRUM_FRAC_LOW * (val - _SPECTRUM_DBFS_LO) / (_SPECTRUM_DBFS_SPLIT1 - _SPECTRUM_DBFS_LO)
    elif val <= _SPECTRUM_DBFS_SPLIT2:
        return _SPECTRUM_FRAC_LOW + _SPECTRUM_FRAC_MID * (val - _SPECTRUM_DBFS_SPLIT1) / (_SPECTRUM_DBFS_SPLIT2 - _SPECTRUM_DBFS_SPLIT1)
    else:
        frac_hi = 1.0 - _SPECTRUM_FRAC_LOW - _SPECTRUM_FRAC_MID
        return _SPECTRUM_FRAC_LOW + _SPECTRUM_FRAC_MID + frac_hi * (val - _SPECTRUM_DBFS_SPLIT2) / (_SPECTRUM_DBFS_HI - _SPECTRUM_DBFS_SPLIT2)


def draw_panel(surface, fonts, surf_cache, title, val, px, pw, ph, lr_vals=None):
    font_title, font_scale, font_value, font_lr, *_ = fonts

    TITLE_H = 85
    PAD     = 6
    MARGIN  = 10
    LO, HI  = -60.0, 0.0

    BAR_Y = TITLE_H
    BAR_X = px + PAD
    BAR_W = pw - PAD * 2
    BAR_H = ph - TITLE_H - PAD

    color = lufs_to_color(val)

    # Title (cached — never changes)
    surface.set_clip(pygame.Rect(px, 0, pw, TITLE_H))
    title_key = ('title', title)
    if title_key not in surf_cache:
        surf_cache[title_key] = font_title.render(title, True, (170, 170, 170))
    t = surf_cache[title_key]
    surface.blit(t, (px + pw // 2 - t.get_width() // 2, 8))
    surface.set_clip(None)

    # Bar background
    pygame.draw.rect(surface, (28, 28, 28), (BAR_X, BAR_Y, BAR_W, BAR_H))

    # Bar fill
    fill_h = int(BAR_H * _y_ratio(val, LO, HI))
    pygame.draw.rect(surface, color, (BAR_X, BAR_Y + BAR_H - fill_h, BAR_W, fill_h))

    # -23 LUFS 標準線（加粗，不顯示其他刻度）
    ty = BAR_Y + BAR_H - int(BAR_H * _y_ratio(TARGET_LUFS, LO, HI))
    pygame.draw.line(surface, (255, 255, 255), (BAR_X, ty), (BAR_X + BAR_W, ty), 4)

    # 大數字 — 標準線以上區域，垂直置中
    num_zone_h = ty - BAR_Y
    sc = _fit_scale(font_value, _LUFS_REFS, 34, pw - MARGIN * 2, num_zone_h - MARGIN * 2)
    surface.set_clip(pygame.Rect(px, BAR_Y, pw, num_zone_h))
    draw_glyph_text(surface, surf_cache, font_value, _fmt_lufs(val), (255, 255, 255), 34, sc, _WHITE_ONLY,
                    px + pw // 2, BAR_Y + num_zone_h // 2)
    surface.set_clip(None)

    # L/R — 標準線以下區域，垂直置中，白色，格式 "L -23"（不縮放，原字級）
    if lr_vals is not None:
        lr_h      = font_lr.get_height() + 17 * 2
        lr_zone_y = ty
        lr_zone_h = BAR_Y + BAR_H - ty
        gap       = 12
        lr_start  = lr_zone_y + lr_zone_h // 2 - (lr_h * 2 + gap) // 2
        surface.set_clip(pygame.Rect(BAR_X, lr_zone_y, BAR_W, lr_zone_h))
        for k, (prefix, v) in enumerate(zip(("L ", "R "), lr_vals)):
            draw_glyph_text(surface, surf_cache, font_lr, _fmt_lufs(v, prefix), (255, 255, 255), 17, 1, _WHITE_ONLY,
                            px + pw // 2, lr_start + k * (lr_h + gap) + lr_h // 2)
        surface.set_clip(None)


def draw_number_only_panel(surface, fonts, surf_cache, title, val, px, pw, py, ph, delta=None, delta_align='below', num_color=(255, 255, 255), val_str=None, ref_strs=_LUFS_REFS):
    font_title, _, font_value, font_lr, font_delta = fonts

    TITLE_H        = 70
    _pscale        = pw / 640           # 640 = 1920//3，以此推算解析度比例
    OUTLINE_OFFSET = max(4, round(34 * _pscale))
    MARGIN         = 10

    surface.set_clip(pygame.Rect(px, py, pw, ph))

    title_key = ('title', title, pw)   # 含 pw：draw_panel 也用 ('title', title)，縮小過的版本不能被它拿去用
    if title_key not in surf_cache:
        t = font_title.render(title, True, (170, 170, 170))
        if t.get_width() > pw - 8:     # 標題比格寬長（例如左欄的 SHORT TERM (3")）就等比縮小
            t = pygame.transform.smoothscale(t, (pw - 8, max(1, t.get_height() * (pw - 8) // t.get_width())))
        surf_cache[title_key] = t
    t = surf_cache[title_key]
    surface.blit(t, (px + pw // 2 - t.get_width() // 2, py + 10))

    num_h   = ph - TITLE_H
    if val_str is None:
        val_str = _fmt_lufs(val)
    sc = _fit_scale(font_value, ref_strs, OUTLINE_OFFSET, pw - MARGIN * 2, num_h - MARGIN * 2)
    vx, vy, new_w, new_h = draw_glyph_text(surface, surf_cache, font_value, val_str, num_color, OUTLINE_OFFSET, sc,
                                           _NUM_PALETTE, px + pw // 2, py + TITLE_H + num_h // 2)

    if delta is not None:
        d_str   = str(delta)
        d_color = (220, 50, 50) if delta > 0 else ((110, 200, 110) if delta < 0 else (140, 140, 140))
        delta_half_h = (font_delta.get_height() + 17 * 2) // 2

        content_left = vx + int(OUTLINE_OFFSET * sc)
        minus_w_sc   = int(font_value.size('-')[0] * sc)
        delta_x      = content_left + minus_w_sc // 2

        if delta_align == 'side':
            delta_center_y = py + round(ph * 0.32)
        else:
            num_bottom     = vy + new_h
            delta_center_y = min(py + ph - delta_half_h - 6,
                                 num_bottom + delta_half_h + 4)

        draw_glyph_text(surface, surf_cache, font_delta, d_str, d_color, 17, 1, _DELTA_PALETTE, delta_x, delta_center_y)

    surface.set_clip(None)


def compute_spectrum_bands():
    """取 buf_s_chunks 尾端 SPECTRUM_FFT_SIZE 樣本做 FFT，回傳每個 1/3 Octave 頻段校準過的 dBFS。
    只在 display_mode == "spectrum" 時才會被呼叫，不背景常駐、不額外開執行緒。
    頻段內用「加總」不用「平均」——加總才符合 IEC 61260 風格頻段能量的物理意義，
    也才能讓已知電平的測試音（PM5D 振盪器）準確對上讀值，平均會被空 bin 稀釋。
    左右聲道「各自」做 FFT 算功率後才取平均，不能先在時域把兩聲道加在一起再做 FFT——
    兩聲道相位沒對齊時（例如某聲道熱冷線接反）時域平均會互相抵消，功率平均不會，
    且兩聲道電平/相位正常時數值跟時域平均完全一致，校準基準不用變。"""
    snap = list(buf_s_chunks)
    if not snap:
        return [_SPECTRUM_DBFS_LO] * len(OCTAVE_BANDS)
    audio = np.concatenate(snap)
    if len(audio) < SPECTRUM_FFT_SIZE:
        return [_SPECTRUM_DBFS_LO] * len(OCTAVE_BANDS)

    window = audio[-SPECTRUM_FFT_SIZE:]
    power_avg = np.zeros(SPECTRUM_FFT_SIZE // 2 + 1)
    for ch in range(CHANNELS):
        spectrum = np.fft.rfft(window[:, ch] * _HANN_WINDOW)
        power_avg += np.abs(spectrum) ** 2
    power_avg /= CHANNELS

    levels_dbfs = []
    for fc, bin_lo, bin_hi in OCTAVE_BANDS:
        power = float(np.sum(power_avg[bin_lo:bin_hi]))
        levels_dbfs.append(10 * np.log10(power / _DBFS_REF + 1e-12))
    return levels_dbfs


def _update_spectrum_peaks(band_levels):
    """Peak hold：新值比目前 peak 高就直接跳上去，並重新頂住 _PEAK_HOLD_FRAMES 幀；
    頂住期間完全不落下，時間到了才開始依固定速度慢慢降。
    只在原地改 _spectrum_peaks／_spectrum_peak_hold 內容，不用 global、不用鎖。"""
    for i, v in enumerate(band_levels):
        if v > _spectrum_peaks[i]:
            _spectrum_peaks[i]     = v
            _spectrum_peak_hold[i] = _PEAK_HOLD_FRAMES
        elif _spectrum_peak_hold[i] > 0:
            _spectrum_peak_hold[i] -= 1
        else:
            _spectrum_peaks[i] = max(v, _spectrum_peaks[i] - _PEAK_DECAY_PER_FRAME)


_AXIS_ALPHA = 153   # 軸文字 60% 不透明：字壓在 bar 上時 bar 的顏色透得出來


def _axis_label(surf_cache, font, text):
    """軸文字：灰字＋2px 黑描邊＋60% 透明（整張圖 set_alpha，字與描邊一起變淡）。字串固定、張數有限，照常快取。"""
    key = ('axis', text)
    lab = surf_cache.get(key)
    if lab is None:
        lab = render_outlined(font, text, (150, 150, 150), offset=2)
        lab.set_alpha(_AXIS_ALPHA)
        surf_cache[key] = lab
    return lab


def _draw_dbfs_ticks(surface, surf_cache, font_axis, px, pw, bar_zone_y, bar_zone_h, margin):
    """Y 軸參考線與兩側刻度文字。Spectrum 與 Histogram 共用同一份——兩張圖跟長條用同一條
    _spectrum_y_ratio 三段式曲線算位置（中段密、兩端疏），共用才不可能對不齊。
    最底的 -70 只畫線不印字：圖底本身就代表 -70，而且那個角落要留給 X 軸文字（字大了會疊在一起）。"""
    bar_area_w = pw - margin * 2
    for tick in _SPECTRUM_DBFS_TICKS:
        ty = bar_zone_y + bar_zone_h - int(bar_zone_h * _spectrum_y_ratio(tick))
        pygame.draw.line(surface, (70, 70, 70), (px + margin, ty), (px + margin + bar_area_w, ty), 1)
        if tick == _SPECTRUM_DBFS_LO:
            continue
        label   = _axis_label(surf_cache, font_axis, str(tick))
        label_y = max(bar_zone_y, min(bar_zone_y + bar_zone_h - label.get_height(), ty - label.get_height() // 2))
        surface.blit(label, (px + margin + 2, label_y))
        surface.blit(label, (px + pw - margin - label.get_width() - 2, label_y))


def _draw_vguides(surface, xs, top, bottom):
    """垂直輔助線（同水平格線的顏色）。要在畫 bar 之前呼叫——被 bar 擋住的部分就看不到，只在 bar 上方露出。"""
    for x in xs:
        pygame.draw.line(surface, (70, 70, 70), (x, top), (x, bottom - 1), 1)


def _draw_x_labels(surface, surf_cache, font_axis, items, px, pw, bottom, margin):
    """X 軸文字直接壓在圖底、bar 上面（有描邊與透明度，不另留一列——兩張圖的高度因此完全一致）。
    items = [(x 中心, 文字)]；頭尾標籤夾在圖框內不出界。"""
    for x, text in items:
        lab = _axis_label(surf_cache, font_axis, text)
        lx  = max(px + margin, min(x - lab.get_width() // 2, px + pw - margin - lab.get_width()))
        surface.blit(lab, (lx, bottom - lab.get_height() - 2))


def draw_spectrum_panel(surface, fonts, surf_cache, band_levels_dbfs, px, pw, py, ph, highlight_index=None, peak_levels=None):
    """全螢幕（扣掉左側資訊欄）1/3 Octave 長條圖，長條區左右兩側都畫 dBFS Y 軸刻度。"""
    font_title, font_scale = fonts[0], fonts[1]
    TITLE_H = 50
    MARGIN  = 4

    title_key = ('spectrum_title',)
    if title_key not in surf_cache:
        surf_cache[title_key] = font_title.render("SPECTRUM ANALYZER", True, (170, 170, 170))
    t = surf_cache[title_key]
    surface.blit(t, (px + pw // 2 - t.get_width() // 2, py + 8))

    n = len(band_levels_dbfs)
    if n == 0:
        return
    bar_zone_y = py + TITLE_H
    bar_zone_h = ph - TITLE_H - MARGIN
    bar_area_w = pw - MARGIN * 2
    bar_w      = max(2, (bar_area_w - MARGIN * (n - 1)) // n)

    bottom     = bar_zone_y + bar_zone_h
    x_center   = [px + MARGIN + i * (bar_w + MARGIN) + bar_w // 2 for i in range(n)]
    x_labels   = [(x_center[i], text) for i, text in _SPECTRUM_XLABELS if i < n]

    pygame.draw.rect(surface, (28, 28, 28), (px + MARGIN, bar_zone_y, bar_area_w, bar_zone_h))
    _draw_vguides(surface, [x for x, _ in x_labels], bar_zone_y, bottom)

    for i, dbfs in enumerate(band_levels_dbfs):
        bx       = px + MARGIN + i * (bar_w + MARGIN)
        ratio    = _spectrum_y_ratio(dbfs)
        bh       = int(bar_zone_h * ratio)
        by       = bottom - bh
        selected = highlight_index is not None and i == highlight_index
        pygame.draw.rect(surface, (90, 170, 230) if selected else (60, 140, 90), (bx, by, bar_w, bh))

        if peak_levels is not None:
            peak_y = bottom - int(bar_zone_h * _spectrum_y_ratio(peak_levels[i]))
            if selected:    # 選定頻段的 peak：黃色加粗，一眼找得到（BYPASS 時 highlight_index 不會落在 0..n-1）
                pygame.draw.rect(surface, (220, 200, 50), (bx, peak_y - 2, bar_w, 4))
            else:
                pygame.draw.rect(surface, (230, 230, 230), (bx, peak_y - 1, bar_w, 2))

    _draw_dbfs_ticks(surface, surf_cache, font_scale, px, pw, bar_zone_y, bar_zone_h, MARGIN)
    _draw_x_labels(surface, surf_cache, font_scale, x_labels, px, pw, bottom, MARGIN)


# Spectrum X 軸標示的頻段（使用者指定，非平均分佈；10k 跟 6.3k 只差 2 根、40px 字會疊在一起，所以沒列）
_SPECTRUM_XLABELS = [(i, f"{fc / 1000:g}k" if fc >= 1000 else f"{fc:g}")
                     for i, (fc, _, _) in enumerate(OCTAVE_BANDS) if fc in (80, 400, 1000, 6300)]

# X 軸刻度（位置 = 佔視窗的比例，文字）：整點版每 10 分鐘一格；滑動版每 1 分鐘一格（3' / 2' / 1' / now）
_TL_XTICKS = {
    "hour":  [(m / 60, str(m)) for m in range(0, 61, 10)],
    "slide": [((_TL_SLIDE_MIN - m) / _TL_SLIDE_MIN, f"{m}'" if m else "now") for m in range(_TL_SLIDE_MIN, -1, -1)],
}


def draw_histogram_panel(surface, fonts, surf_cache, mode, snap, title, px, pw, py, ph):
    """時間軸響度歷史圖，骨架同 draw_spectrum_panel。Y 軸與 Spectrum 共用同一條曲線與刻度，長條顏色
    照 lufs_to_color，-23 目標線畫成水平線。整點版 508 欄 × 2px（≈ 7 秒／欄，相鄰不留縫）；滑動版
    60 欄 × 約 17px（3 秒／欄，留 2px 縫，一格一格看得出來）。取樣約每 1.14 秒一筆，欄內取最大值。
    欄邊界用 i * bar_area_w // n 算，寬度除不盡時自然 16 / 17px 交錯，不必湊整除。"""
    font_title, font_scale = fonts[0], fonts[1]
    TITLE_H = 50
    MARGIN  = 4
    t0, span, t, s, _ = snap

    title_key = ('hist_title', title)
    if title_key not in surf_cache:
        surf_cache[title_key] = font_title.render(title, True, (170, 170, 170))
    tt = surf_cache[title_key]
    surface.blit(tt, (px + pw // 2 - tt.get_width() // 2, py + 8))

    bar_zone_y = py + TITLE_H
    bar_zone_h = ph - TITLE_H - MARGIN            # 跟 Spectrum 同高：X 軸文字壓在圖底，不另留一列
    bar_area_w = pw - MARGIN * 2
    n, gap     = (bar_area_w // 2, 0) if mode == "hour" else (_TL_SLIDE_N, 2)
    bottom     = bar_zone_y + bar_zone_h
    x_ticks    = [(px + MARGIN + int(frac * bar_area_w), text) for frac, text in _TL_XTICKS[mode]]

    pygame.draw.rect(surface, (28, 28, 28), (px + MARGIN, bar_zone_y, bar_area_w, bar_zone_h))
    # 垂直輔助線：頭尾就是圖框不另畫；-1 讓線落在欄邊界前的那條縫裡（滑動版 bar 之間留 2px 縫）
    _draw_vguides(surface, [x - 1 for x, _ in x_ticks[1:-1]], bar_zone_y, bottom)

    for i, v in enumerate(tl_columns(t, s, t0, span, n)):
        if v != v:                                  # NaN：這一欄沒有樣本
            continue
        bh = int(bar_zone_h * _spectrum_y_ratio(v))
        if bh > 0:                                  # 低於 -70（靜音）不畫，跟 Spectrum 的底一致
            x0 = i * bar_area_w // n
            x1 = (i + 1) * bar_area_w // n
            pygame.draw.rect(surface, lufs_to_color(v), (px + MARGIN + x0, bottom - bh, x1 - x0 - gap, bh))

    _draw_dbfs_ticks(surface, surf_cache, font_scale, px, pw, bar_zone_y, bar_zone_h, MARGIN)

    # -23 LUFS 目標線：判讀重點就是「響度有沒有貼在目標線上」
    ty = bottom - int(bar_zone_h * _spectrum_y_ratio(TARGET_LUFS))
    pygame.draw.line(surface, (255, 255, 255), (px + MARGIN, ty), (px + MARGIN + bar_area_w, ty), 3)

    _draw_x_labels(surface, surf_cache, font_scale, x_ticks, px, pw, bottom, MARGIN)


def draw_histogram_screen(surface, fonts, surf_cache, W, H, mode, hour_now, vals):
    """Histogram 整個畫面：左側三格 MOMENTARY / SHORT TERM (3") / LRA（標題寫法同 Loudness 畫面），
    右側時間軸歷史圖，版面同 Spectrum 畫面。"""
    left_w  = W // 5
    third_h = H // 3
    snap    = tl_snapshot(mode)
    lra_s   = "---" if snap[4] is None else f"{snap[4]:.1f}"

    draw_number_only_panel(surface, fonts, surf_cache, "MOMENTARY", vals["M"], 0, left_w, 0, third_h,
                           num_color=lufs_to_color(vals["M"]))
    draw_number_only_panel(surface, fonts, surf_cache, 'SHORT TERM (3")', vals["S"], 0, left_w, third_h, third_h,
                           num_color=lufs_to_color(vals["S"]))
    draw_number_only_panel(surface, fonts, surf_cache, "LRA", None, 0, left_w, 2 * third_h, H - 2 * third_h,
                           val_str=lra_s, num_color=(255, 255, 255), ref_strs=_LRA_REFS)
    for i in (1, 2):
        pygame.draw.line(surface, (55, 55, 55), (0, i * third_h), (left_w, i * third_h), 1)
    pygame.draw.line(surface, (55, 55, 55), (left_w, 0), (left_w, H), 1)

    title = f"LUFS HISTOGRAM - THIS HOUR ({hour_now})" if mode == "hour" else f"LUFS HISTOGRAM - SLIDING ({_TL_SLIDE_MIN}')"
    draw_histogram_panel(surface, fonts, surf_cache, mode, snap, title, left_w, W - left_w, 0, H)


def draw_loudness_screen(surface, fonts, surf_cache, W, H, vals, hour_now, prev_H, prev_SEG):
    """Loudness 畫面：MOMENTARY／SHORT TERM 兩格大 BAR，右欄 THIS HOUR（上）＋ SEGMENT（下）。"""
    pw = W // 3
    for idx, (title, val, lr) in enumerate([
        ("MOMENTARY",       vals["M"], (vals["M_L"], vals["M_R"])),
        ('SHORT TERM (3")', vals["S"], None),
    ]):
        draw_panel(surface, fonts, surf_cache, title, val, idx * pw, pw, H, lr_vals=lr)

    half_h    = H // 2
    h_delta   = round(vals["H"]) - round(prev_H)     if prev_H   is not None and vals["H"]   > -70 else None
    seg_delta = round(vals["SEG"]) - round(prev_SEG) if prev_SEG is not None and vals["SEG"] > -70 else None
    draw_number_only_panel(surface, fonts, surf_cache, f"THIS HOUR ({hour_now})", vals["H"], 2 * pw, pw, 0, half_h,
                           delta=h_delta, num_color=lufs_to_color(vals["H"]))
    pygame.draw.line(surface, (55, 55, 55), (2 * pw, half_h), (2 * pw + pw, half_h), 1)
    draw_number_only_panel(surface, fonts, surf_cache, "SEGMENT (3')", vals["SEG"], 2 * pw, pw, half_h, H - half_h,
                           delta=seg_delta, delta_align='side')
    for i in (1, 2):
        pygame.draw.line(surface, (55, 55, 55), (i * pw, 0), (i * pw, H), 1)


def draw_spectrum_screen(surface, fonts, surf_cache, W, H, vals, band_levels, peaks, freq_idx):
    """Spectrum 畫面：左欄 ALL LUFS／BAND dBFS／BAND FREQ，右側 1/3 Octave 長條圖。"""
    left_w = W // 5
    if freq_idx == BYPASS_INDEX:
        freq_str, dbfs_str = "ALL", "---"
    else:
        freq_str = _fmt_freq(OCTAVE_BANDS[freq_idx][0])
        dbfs_str = f"{round(peaks[freq_idx])}"

    third_h = H // 3
    draw_number_only_panel(surface, fonts, surf_cache, "ALL LUFS", vals["M"], 0, left_w, 0, third_h,
                           num_color=lufs_to_color(vals["M"]))
    draw_number_only_panel(surface, fonts, surf_cache, "BAND dBFS", None, 0, left_w, third_h, third_h,
                           val_str=dbfs_str, num_color=(255, 255, 255))
    draw_number_only_panel(surface, fonts, surf_cache, "BAND FREQ", None, 0, left_w, 2 * third_h, H - 2 * third_h,
                           val_str=freq_str, num_color=(255, 255, 255), ref_strs=_FREQ_REFS)
    for i in (1, 2):
        pygame.draw.line(surface, (55, 55, 55), (0, i * third_h), (left_w, i * third_h), 1)
    pygame.draw.line(surface, (55, 55, 55), (left_w, 0), (left_w, H), 1)

    draw_spectrum_panel(surface, fonts, surf_cache, band_levels, left_w, W - left_w, 0, H,
                        highlight_index=freq_idx, peak_levels=peaks)


_DISPLAY_CYCLE = ("loudness", "spectrum", "histogram")


def _toggle_display_mode():
    global display_mode
    display_mode = _DISPLAY_CYCLE[(_DISPLAY_CYCLE.index(display_mode) + 1) % len(_DISPLAY_CYCLE)]
    print(f"[GPIO DEBUG] toggle fired (GPIO{GPIO_PIN_TOGGLE}) -> display_mode={display_mode}", flush=True)


def _step_freq(delta):
    """上一個/下一個頻率鈕的共用處理——Spectrum 畫面步進頻率；Histogram 畫面則是兩顆都切換
    時間範圍（整點起 0~60 分 ↔ 最近 10 分鐘，只有兩個選項，方向沒有意義）；Loudness 畫面不生效，避免 PM5D 端誤觸。
    撞到邊界時繞到另一頭（wraparound），不是卡住不動：萬一其中一顆鍵故障，
    另一顆單獨連續按繞一圈還是能到達所有頻段，不會有走不到的死角。
    清單最後一格（BYPASS_INDEX）代表「不選定特定頻段」，純顯示狀態（BAND dBFS／
    BAND FREQ 顯示 ---／ALL），不影響任何音訊處理——Pi 本來就不輸出音訊。
    純粹更新 freq_index，不做任何運算，停在哪一格完全零成本。"""
    global freq_index, hist_mode
    pin = GPIO_PIN_FREQ_PREV if delta < 0 else GPIO_PIN_FREQ_NEXT
    if display_mode == "histogram":
        hist_mode = "slide" if hist_mode == "hour" else "hour"
        print(f"[GPIO DEBUG] step_freq fired (GPIO{pin}, delta={delta}) -> hist_mode={hist_mode}", flush=True)
        return
    if display_mode != "spectrum":
        print(f"[GPIO DEBUG] step_freq fired (GPIO{pin}, delta={delta}) -> ignored, not in spectrum mode", flush=True)
        return
    freq_index = (freq_index + delta) % N_FREQ_STEPS
    if freq_index == BYPASS_INDEX:
        print(f"[GPIO DEBUG] step_freq fired (GPIO{pin}, delta={delta}) -> freq_index={freq_index}, BYPASS（不選定頻段）", flush=True)
        return
    f_center = OCTAVE_BANDS[freq_index][0]
    print(f"[GPIO DEBUG] step_freq fired (GPIO{pin}, delta={delta}) -> freq_index={freq_index}, f_center={f_center}Hz", flush=True)


def _log_cache(surf_cache):
    """診斷用（2026-10-03 加）：響度計連跑 14 天 RSS 長到 346 MB，成長全在主 heap（Rust 那邊不動），
    懷疑是 surf_cache 沒有上限——每出現一個新的數字字串就多快取幾張大圖。每 10 分鐘印一行：
    快取張數、總 MB、各類張數，附 RSS。快取 MB 跟著 RSS 一起漲就是它；RSS 漲但快取不動就是別的東西。"""
    by_type = collections.Counter()
    nbytes  = 0
    for k, v in surf_cache.items():
        for s in (v.values() if isinstance(v, dict) else (v,)):   # 'glyphs' 存的是一整套字元的 dict
            nbytes += s.get_width() * s.get_height() * s.get_bytesize()
        by_type[k[0]] += 1
    with open("/proc/self/status") as f:
        rss = next(int(line.split()[1]) for line in f if line.startswith("VmRSS:"))
    print(f"[CACHE] n={len(surf_cache)} {nbytes / 1e6:.1f}MB rss={rss // 1024}MB "
          + " ".join(f"{k}={c}" for k, c in by_type.most_common()), flush=True)


def make_fonts(W):
    """以 1920 為基準做比例縮放，適應任何解析度。tools/ 的離屏渲染與量測腳本也呼叫這裡，字級只有一份。"""
    _scale = W / 1920
    _fs_value  = max(80,  round(250 * _scale))
    _fs_lr     = max(50,  round(140 * _scale))
    _fs_delta  = max(32,  round(100 * _scale))   # delta 獨立字型，比 L/R 略小
    _fs_title  = max(24,  round(70  * _scale))
    _fs_axis   = max(16,  round(60  * _scale))   # Spectrum／Histogram 軸文字（1280 寬時 40px），2026-10-09 由 32 加大

    font_title = pygame.font.SysFont("monospace", _fs_title, bold=True)
    font_axis  = pygame.font.SysFont("monospace", _fs_axis)

    dseg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts", "DSEG7Modern-Bold.ttf")
    try:
        font_value = pygame.font.Font(dseg_path, _fs_value)
        font_lr    = pygame.font.Font(dseg_path, _fs_lr)
        font_delta = pygame.font.Font(dseg_path, _fs_delta)
        print(f"DSEG7Modern loaded: {dseg_path}")
    except (FileNotFoundError, pygame.error) as e:
        print(f"DSEG7Modern not found ({e}), fallback to monospace")
        font_value = pygame.font.SysFont("monospace", _fs_value, bold=True)
        font_lr    = pygame.font.SysFont("monospace", _fs_lr, bold=True)
        font_delta = pygame.font.SysFont("monospace", _fs_delta, bold=True)

    return (font_title, font_axis, font_value, font_lr, font_delta)


def warm_glyph_cache(fonts, surf_cache, W, H):
    """開機預熱：三個畫面（Histogram 兩種模式）各離屏畫一次。每種字級第一次出現時，_glyph_set 會把整套
    字元 × 整個調色盤一次畫好，所以畫一次就涵蓋執行中所有可能的數字與顏色。
    要在開音訊 stream 之前呼叫——這一兩秒的大量渲染如果發生在執行中，就是 2026-10 那 7 次溢位的成因。"""
    warm  = pygame.Surface((W, H))
    dummy = {"M": -23.0, "S": -23.0, "H": -23.0, "SEG": -23.0, "M_L": -23.0, "M_R": -23.0}
    draw_loudness_screen(warm, fonts, surf_cache, W, H, dummy, 0, -24.0, -24.0)      # 給前值，變化量才會畫到
    draw_spectrum_screen(warm, fonts, surf_cache, W, H, dummy,
                         [_SPECTRUM_DBFS_LO] * len(OCTAVE_BANDS), _spectrum_peaks, 0)
    for mode in ("hour", "slide"):
        draw_histogram_screen(warm, fonts, surf_cache, W, H, mode, 0, dummy)


def main():
    global current_hour, reset_hour_flag, _running, prev_H, prev_SEG

    DEVICE = select_input_device(sd.query_devices(), os.environ.get("LM_AUDIO_DEVICE", "").strip(),
                                 interactive=sys.stdin.isatty())

    # Start compute thread before opening audio stream
    ct = threading.Thread(target=compute_loop, daemon=True)
    ct.start()

    # pygame.init() 預設會連 mixer 一起初始化，佔用 ALSA 音訊裝置，曾經跟 sounddevice
    # 搶同一台 Scarlett 導致 stream 開不起來。這支程式完全不用 pygame 播音訊，用 dummy
    # driver 讓 pygame 不碰真實音訊硬體即可解決，防禦性保留（零成本，不影響任何行為）。
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    pygame.init()
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    W, H   = screen.get_size()
    pygame.mouse.set_visible(False)
    clock  = pygame.time.Clock()

    fonts = make_fonts(W)

    # surface cache：標題、軸文字、單一字元（各字級各色）——開機預熱後就不會再長
    surf_cache = {}

    seg_snapshot_time = time.time() + 180.0   # 3 分鐘後第一次快照
    cache_log_time    = time.time() + 60.0    # 快取診斷 log：1 分鐘後第一筆，之後每 10 分鐘

    btn_toggle = Button(GPIO_PIN_TOGGLE, pull_up=True, bounce_time=0.05)
    btn_toggle.when_pressed = _toggle_display_mode

    btn_freq_prev = Button(GPIO_PIN_FREQ_PREV, pull_up=True, bounce_time=0.05)
    btn_freq_prev.when_pressed = lambda: _step_freq(-1)

    btn_freq_next = Button(GPIO_PIN_FREQ_NEXT, pull_up=True, bounce_time=0.05)
    btn_freq_next.when_pressed = lambda: _step_freq(1)

    # 錄影/遠端除錯用：不用碰實體按鍵，SSH 執行 `kill -USR1 <PID>`／`kill -USR2 <PID>`
    # 就能觸發下一個/上一個頻率，直接呼叫跟按鍵回呼相同的 _step_freq()。
    # SIGUSR1/SIGUSR2 都用掉了，畫面切換借用 SIGHUP（`kill -HUP <PID>`）——這支程式
    # 不依賴終端機、沒有需要處理「真正 hangup」的情境，借用完全安全。
    signal.signal(signal.SIGUSR1, lambda sig, frame: _step_freq(1))
    signal.signal(signal.SIGUSR2, lambda sig, frame: _step_freq(-1))
    signal.signal(signal.SIGHUP, lambda sig, frame: _toggle_display_mode())

    # 單工輸入 stream——耳機監聽改回 Scarlett 硬體 Direct Monitor（零延遲），
    # Pi 只做分析／顯示，不輸出任何音訊。
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, device=DEVICE,
        channels=CHANNELS, dtype="float32",
        blocksize=BLOCK_SIZE, callback=audio_callback,
    )

    t_warm = time.monotonic()
    warm_glyph_cache(fonts, surf_cache, W, H)       # 一定要在 `with stream:` 之前，見 docstring
    print(f"glyph cache warmed: {len(surf_cache)} entries in {time.monotonic() - t_warm:.2f}s", flush=True)

    with stream:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    _running = False
                    pygame.quit(); sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    _running = False
                    pygame.quit(); sys.exit()

            now_h = datetime.datetime.now().hour
            if now_h != current_hour:
                with lock:
                    h_val = latest["H"]
                prev_H = h_val if h_val > -70 else None   # 存前一小時值
                current_hour = now_h
                reset_hour_flag = True

            now_t = time.time()
            if now_t >= seg_snapshot_time:
                with lock:
                    seg_val = latest["SEG"]
                if seg_val > -70:
                    prev_SEG = seg_val
                seg_snapshot_time = now_t + 180.0

            if now_t >= cache_log_time:
                _log_cache(surf_cache)
                cache_log_time = now_t + 600.0

            screen.fill((15, 15, 15))

            with lock:
                vals = dict(latest)

            if display_mode == "loudness":
                draw_loudness_screen(screen, fonts, surf_cache, W, H, vals, current_hour, prev_H, prev_SEG)
            elif display_mode == "histogram":
                draw_histogram_screen(screen, fonts, surf_cache, W, H, hist_mode, current_hour, vals)
            else:
                band_levels = compute_spectrum_bands()
                _update_spectrum_peaks(band_levels)
                draw_spectrum_screen(screen, fonts, surf_cache, W, H, vals, band_levels, _spectrum_peaks, freq_index)

            pygame.display.flip()
            # 主執行緒每幀成本（fill／flip／數字格）隨幀率成比例，降幀率是最便宜的降載手段。
            # 代價：KEY 切換的畫面反應最長慢一幀（Loudness 100ms、Histogram 200ms）
            clock.tick(_SCREEN_FPS[display_mode])


if __name__ == "__main__":
    main()
