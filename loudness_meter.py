import numpy as np
import sounddevice as sd
import pygame
import sys
import subprocess
import threading
import collections
import datetime
import time
import os
import signal
from scipy.signal import lfilter
from gpiozero import Button
from lufs_core import EnergyStore   # Rust 擴充模組（lufs_core/），保管 block 能量並負責 H / SEG 的 gating 運算

# ─── Config ───────────────────────────────────────────────────────────────────
SAMPLE_RATE = 48000
CHANNELS    = 2

def _setup_scarlett():
    """
    Auto-detect Focusrite Scarlett. If not found, list all input devices and
    prompt the user to select one.
    """
    devices = sd.query_devices()

    # 優先自動偵測 Scarlett / Focusrite
    scarlett_idx = next(
        (i for i, d in enumerate(devices)
         if ('scarlett' in d['name'].lower() or 'focusrite' in d['name'].lower())
         and d['max_input_channels'] > 0),
        None
    )
    if scarlett_idx is not None:
        print(f"Scarlett detected: [{scarlett_idx}] {devices[scarlett_idx]['name']}")
        return scarlett_idx

    # 找不到 Scarlett → 列出所有可用輸入裝置
    input_devices = [(i, d) for i, d in enumerate(devices) if d['max_input_channels'] > 0]
    if not input_devices:
        print("ERROR: No input devices found.")
        sys.exit(1)

    print("Scarlett not found. Available input devices:")
    for n, (i, d) in enumerate(input_devices, 1):
        print(f"  {n}. {d['name']}  (ch: {d['max_input_channels']})")

    while True:
        try:
            choice = int(input(f"Select device [1-{len(input_devices)}]: "))
            if 1 <= choice <= len(input_devices):
                idx, dev = input_devices[choice - 1]
                print(f"Using: [{idx}] {dev['name']}")
                return idx
        except (ValueError, EOFError):
            pass
        print(f"  Please enter a number between 1 and {len(input_devices)}.")

BLOCK_SIZE  = 2400          # 50ms per block

TARGET_LUFS      = -23.0
BLUE_THRESHOLD   = -30.0
YELLOW_THRESHOLD = -23.0
RED_THRESHOLD    = -10.0
FPS              = 20

# 窗口長度以「幾個 block」表示（BLOCK_SIZE = 50ms），直接決定 deque 的 maxlen
M_BLOCKS = 8    # 瞬時窗口 400ms
S_BLOCKS = 60   # 短期窗口 3s

# H / SEG 的相對閾值：ITU-R BS.1770-5 Annex 1 官方值是「絕對閾值篩選後平均再 −10 LU」。
# 2026-09-19 前這裡誤植成 −8（2026-03-27 手寫 compute_h() 取代 pyloudnorm 時的錄入錯誤），
# 大動態範圍內容的 H 因此偏高最多約 0.46 LU，見 implementation-notes.md。
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

_PEAK_HOLD_SEC        = 1.0    # peak 頂住不落下的時間（秒），上機用眼睛調
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
dist_mode    = "hour"       # histogram 畫面的統計範圍："hour" 整點累積 / "slide" 滑動 3 分鐘（KEY10/11 切換）

# Peak hold：每個頻段一格，只在原地用索引更新內容，不重新賦值整個 list，不用 global。
_spectrum_peaks     = [_SPECTRUM_DBFS_LO] * len(OCTAVE_BANDS)
_spectrum_peak_hold = [0] * len(OCTAVE_BANDS)   # 每個頻段「還要再頂住幾幀」的倒數計時

prev_H   = None   # last completed hour's LUFS（整點 rollover 時更新）
prev_SEG = None   # last 3-min fixed segment's LUFS（每 3 分鐘快照）

st_history  = collections.deque(maxlen=20)  # SHORT TERM 每 3 秒快照
seg_history = collections.deque(maxlen=20)  # SEGMENT 每 3 分鐘快照


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


# ─── LUFS Histogram（響度分布；S 每 0.1 秒投一票，記票式累加） ───────────────────
# 內部解析度 0.1 LU（libebur128 同款），P10/P95 誤差 ≤ 0.05 LU。
# 第 0 格 =「≤ -70 LUFS」（低於絕對閾值：不參與 LRA、畫面上併入最左根），第 1..800 格 = -70..+10 LUFS。
_DIST_PER_LU    = 10
_DIST_N         = 801
_DIST_BAR_LO    = -50      # 畫面最左根的中心 LUFS；41 根、每根 1 LU → -50..-10
_DIST_N_BARS    = 41
_DIST_BAR_BIN0  = 1 + round((_DIST_BAR_LO - 0.5 + 70.0) * _DIST_PER_LU)   # 最左根對應的第一個內部格（=196）
_DIST_SLIDE_SEC = 180.0    # 滑動版窗口，對齊 SEGMENT (3')
_DIST_MIN_N     = 600      # 有效票數 < 60 秒份不算 LRA（剛清零時的數字沒有意義）
_DIST_FRAC_LO   = 1e-5     # Y 軸（對數）底端 = 0.001%

_DIST_CENTERS = -70.0 + (np.arange(_DIST_N) - 0.5) / _DIST_PER_LU   # 第 i 格的中心 LUFS（第 0 格不使用）
_DIST_ENERGY  = 10.0 ** ((_DIST_CENTERS + 0.691) / 10.0)            # 對應能量，算相對閾值用

_dist_hour      = np.zeros(_DIST_N, dtype=np.int64)   # 整點累積，整點清零
_dist_slide     = np.zeros(_DIST_N, dtype=np.int64)   # 滑動 3 分鐘
_dist_slide_q   = collections.deque()                 # (時間, 格號)，票過期時據此退票
_dist_hour_seen = current_hour


def _dist_bin(s):
    if s <= -70.0:
        return 0
    return min(_DIST_N - 1, 1 + int((s + 70.0) * _DIST_PER_LU))


def _dist_feed(s):
    """compute_loop 每 0.1 秒呼叫一次，投一票。整點版：+1，整點清零。滑動版：新票 +1、
    過期票 -1（進出帳）。
    ponytail: 滑動版不要「每幀把窗口內的值重掃分桶」——那是 O(n) 全量重算，就是 compute_h
    當年造成每小時週期性掉音訊的同一個錯誤。過期用時間判斷（不是固定票數），因為
    compute_loop 實際頻率略低於 10Hz，固定 1800 票會變成 3 分 20 秒。"""
    global _dist_hour_seen
    idx = _dist_bin(s)
    now = time.monotonic()
    with lock:
        if current_hour != _dist_hour_seen:      # 整點：清空整點累積
            _dist_hour[:] = 0
            _dist_hour_seen = current_hour
        _dist_hour[idx]  += 1
        _dist_slide[idx] += 1
        _dist_slide_q.append((now, idx))
        while now - _dist_slide_q[0][0] > _DIST_SLIDE_SEC:
            _dist_slide[_dist_slide_q.popleft()[1]] -= 1


def _dist_snapshot(mode):
    with lock:
        return (_dist_hour if mode == "hour" else _dist_slide).copy()


def dist_bars(counts):
    """801 個內部格 → 畫面 41 根（每根 1 LU，中心在整數 LUFS）。超出範圍的併入邊緣根，
    「低於閾值」格也併進最左根。"""
    b0   = _DIST_BAR_BIN0
    bars = counts[b0:b0 + _DIST_N_BARS * _DIST_PER_LU].reshape(_DIST_N_BARS, _DIST_PER_LU).sum(axis=1)
    bars[0]  += counts[:b0].sum()
    bars[-1] += counts[b0 + _DIST_N_BARS * _DIST_PER_LU:].sum()
    return bars


def dist_lra(counts):
    """EBU Tech 3342：絕對閾值（-70，即排除第 0 格）→ 相對閾值（過閾值票的平均能量再 -20 LU）
    → P10 / P95。回傳 (LRA, P10, P95)；票數不足回 None。
    相對閾值的基準是「這批 short-term 值自己的平均」，不是 H——舊版（2026-03-27 被刪掉的
    compute_lra）用的是 integrated_lufs - 20，跟 Tech 3342 的定義不同。"""
    c  = counts[1:].astype(np.float64)
    n1 = c.sum()
    if n1 == 0:
        return None
    rel = _energy_to_lufs((c * _DIST_ENERGY[1:]).sum() / n1) - 20.0
    c[_DIST_CENTERS[1:] <= rel] = 0
    n2 = c.sum()
    if n2 < _DIST_MIN_N:
        return None
    cdf = np.cumsum(c)
    p10 = _DIST_CENTERS[1 + int(np.searchsorted(cdf, 0.10 * n2))]
    p95 = _DIST_CENTERS[1 + int(np.searchsorted(cdf, 0.95 * n2))]
    return p95 - p10, p10, p95


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
            _dist_feed(S)   # 窗口還沒填滿 3 秒前的 S 是「沒資料」不是「靜音」，所以只在這裡投票

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


# ─── GUI ──────────────────────────────────────────────────────────────────────
def render_outlined(font, text, color, outline_color=(0, 0, 0), offset=2):
    """2 層渲染：黑色外框（16 方向均勻擴展）→ 主色填充。
    模擬 Photoshop 筆畫「外部」效果：白色主體大小不變，黑框只往外延伸。"""
    import math
    dirs16 = [(math.cos(i * math.pi / 8), math.sin(i * math.pi / 8)) for i in range(16)]

    shadow_surf = font.render(text, True, outline_color)
    color_surf  = font.render(text, True, color)

    w = color_surf.get_width()  + offset * 2
    h = color_surf.get_height() + offset * 2
    surf = pygame.Surface((w, h), pygame.SRCALPHA)

    # 第一層：黑色外框，16 方向往外擴展（雙半徑填補缺口）
    for r in [offset, max(1, round(offset * 0.6))]:
        for dx, dy in dirs16:
            surf.blit(shadow_surf, (round(dx * r) + offset, round(dy * r) + offset))

    # 第二層：白色主體，正中心疊上，大小不變
    surf.blit(color_surf, (offset, offset))
    return surf


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


_HIST_LO  = -41.0   # 歷史色塊高度 0%
_HIST_REF =   0.0   # 歷史色塊高度 100%（0 LUFS = 滿格）


def draw_history_strip(surface, hist_data, zone_x, zone_y, zone_w, zone_h, gap=3):
    """20格 FIFO 歷史色塊：-23 LUFS 為滿格，暗化閾值色 70% 透明度 + 白色描邊。"""
    N       = 20
    n       = len(hist_data)
    block_w = max(4, (zone_w - gap * (N - 1)) // N)
    # 單一 SRCALPHA surface 重複使用，避免每格 allocate 一張新的
    fill_surf = pygame.Surface((block_w, zone_h), pygame.SRCALPHA)
    for i, lufs_val in enumerate(hist_data):
        slot  = N - n + i                        # 右對齊
        bx    = zone_x + slot * (block_w + gap)
        ratio = min(1.0, max(0.0, (lufs_val - _HIST_LO) / (_HIST_REF - _HIST_LO)))
        bh    = max(4, int(zone_h * ratio))
        by    = zone_y + zone_h - bh
        r, g, b = lufs_to_color(lufs_val)
        fill_surf.fill((0, 0, 0, 0))
        fill_surf.fill((r // 2, g // 2, b // 2, 178), (0, zone_h - bh, block_w, bh))
        surface.blit(fill_surf, (bx, zone_y))
        pygame.draw.rect(surface, (255, 255, 255), (bx, by, block_w, bh), 2)


def draw_panel(surface, fonts, surf_cache, title, val, px, pw, ph, lr_vals=None, history=None):
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
    val_str    = _fmt_lufs(val)
    num_key    = ('num_bar', val_str, pw, num_zone_h)
    if num_key not in surf_cache:
        vs_raw = render_outlined(font_value, val_str, (255, 255, 255), offset=34)
        sc     = min((pw - MARGIN * 2) / vs_raw.get_width(),
                     (num_zone_h - MARGIN * 2) / vs_raw.get_height())
        surf_cache[num_key] = pygame.transform.smoothscale(
            vs_raw, (int(vs_raw.get_width() * sc), int(vs_raw.get_height() * sc)))
    vs           = surf_cache[num_key]
    new_w, new_h = vs.get_size()
    vx           = px + pw // 2 - new_w // 2
    vy           = BAR_Y + num_zone_h // 2 - new_h // 2
    surface.set_clip(pygame.Rect(px, BAR_Y, pw, num_zone_h))
    surface.blit(vs, (vx, vy))
    surface.set_clip(None)

    # 歷史色塊：頂端對齊 -23 LUFS 標準線（ty），滿格頂到標準線
    if history is not None and len(history) > 0:
        hist_zone_h = BAR_Y + BAR_H - ty
        draw_history_strip(surface, history,
                           BAR_X + 4, ty + 4,
                           BAR_W - 8, hist_zone_h - 8)

    # L/R — 標準線以下區域，垂直置中，白色，格式 "L -23"
    if lr_vals is not None:
        l_val, r_val = lr_vals
        l_str = _fmt_lufs(l_val, "L ")
        r_str = _fmt_lufs(r_val, "R ")
        for lr_str in (l_str, r_str):
            lr_key = ('lr', lr_str)
            if lr_key not in surf_cache:
                surf_cache[lr_key] = render_outlined(font_lr, lr_str, (255, 255, 255), offset=17)
        l_surf    = surf_cache[('lr', l_str)]
        r_surf    = surf_cache[('lr', r_str)]
        lr_zone_y = ty
        lr_zone_h = BAR_Y + BAR_H - ty
        gap       = 12
        total_h   = l_surf.get_height() + gap + r_surf.get_height()
        lr_start  = lr_zone_y + lr_zone_h // 2 - total_h // 2
        surface.set_clip(pygame.Rect(BAR_X, lr_zone_y, BAR_W, lr_zone_h))
        surface.blit(l_surf, (px + pw // 2 - l_surf.get_width() // 2, lr_start))
        surface.blit(r_surf, (px + pw // 2 - r_surf.get_width() // 2, lr_start + l_surf.get_height() + gap))
        surface.set_clip(None)


def draw_number_only_panel(surface, fonts, surf_cache, title, val, px, pw, py, ph, delta=None, history=None, delta_align='below', num_color=(255, 255, 255), val_str=None):
    font_title, _, font_value, font_lr, font_delta = fonts

    TITLE_H        = 70
    _pscale        = pw / 640           # 640 = 1920//3，以此推算解析度比例
    OUTLINE_OFFSET = max(4, round(34 * _pscale))
    MARGIN         = 10

    if history is not None and len(history) > 0:
        pygame.draw.rect(surface, (15, 15, 15), (px, py, pw, ph))
        draw_history_strip(surface, history, px + 6, py + 6, pw - 12, ph - 12)

    surface.set_clip(pygame.Rect(px, py, pw, ph))

    # title_bg: 80% 不透明暗色背景，讓 title 在歷史色塊上仍可見
    bg_key = ('title_bg', pw, TITLE_H)
    if bg_key not in surf_cache:
        s = pygame.Surface((pw, TITLE_H), pygame.SRCALPHA)
        s.fill((15, 15, 15, 204))
        surf_cache[bg_key] = s
    surface.blit(surf_cache[bg_key], (px, py))

    title_key = ('title', title)
    if title_key not in surf_cache:
        surf_cache[title_key] = font_title.render(title, True, (170, 170, 170))
    t = surf_cache[title_key]
    surface.blit(t, (px + pw // 2 - t.get_width() // 2, py + 10))

    num_h   = ph - TITLE_H
    if val_str is None:
        val_str = _fmt_lufs(val)
    num_key = ('num_panel', val_str, pw, num_h, num_color)
    if num_key not in surf_cache:
        vs_raw = render_outlined(font_value, val_str, num_color, offset=OUTLINE_OFFSET)
        sc_val = min((pw - MARGIN * 2) / vs_raw.get_width(),
                     (num_h - MARGIN * 2) / vs_raw.get_height())
        surf_cache[num_key] = (
            pygame.transform.smoothscale(
                vs_raw, (int(vs_raw.get_width() * sc_val), int(vs_raw.get_height() * sc_val))),
            sc_val,
        )
    vs, sc = surf_cache[num_key]
    new_w, new_h = vs.get_size()
    vx           = px + pw // 2 - new_w // 2
    vy           = py + TITLE_H + num_h // 2 - new_h // 2
    surface.blit(vs, (vx, vy))

    if delta is not None:
        d_str   = str(delta)
        d_color = (220, 50, 50) if delta > 0 else ((110, 200, 110) if delta < 0 else (140, 140, 140))
        d_key   = ('delta', d_str, d_color)
        if d_key not in surf_cache:
            surf_cache[d_key] = render_outlined(font_delta, d_str, d_color, offset=17)
        ds = surf_cache[d_key]

        delta_half_h = ds.get_height() // 2

        content_left = vx + int(OUTLINE_OFFSET * sc)
        minus_w_sc   = int(font_value.size('-')[0] * sc)
        delta_x      = content_left + minus_w_sc // 2

        if delta_align == 'side':
            delta_center_y = py + round(ph * 0.32)
        else:
            num_bottom     = vy + new_h
            delta_center_y = min(py + ph - delta_half_h - 6,
                                 num_bottom + delta_half_h + 4)

        surface.blit(ds, (delta_x - ds.get_width() // 2, delta_center_y - delta_half_h))

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

    pygame.draw.rect(surface, (28, 28, 28), (px + MARGIN, bar_zone_y, bar_area_w, bar_zone_h))

    for i, dbfs in enumerate(band_levels_dbfs):
        bx    = px + MARGIN + i * (bar_w + MARGIN)
        ratio = _spectrum_y_ratio(dbfs)
        bh    = int(bar_zone_h * ratio)
        by    = bar_zone_y + bar_zone_h - bh
        color = (90, 170, 230) if (highlight_index is not None and i == highlight_index) else (60, 140, 90)
        pygame.draw.rect(surface, color, (bx, by, bar_w, bh))

        if peak_levels is not None:
            peak_ratio = _spectrum_y_ratio(peak_levels[i])
            peak_y     = bar_zone_y + bar_zone_h - int(bar_zone_h * peak_ratio)
            pygame.draw.rect(surface, (230, 230, 230), (bx, peak_y - 1, bar_w, 2))

    # Y 軸參考：跟長條圖用同一條三段式曲線算位置，刻度間距會跟著曲線走（中段密、兩端疏）
    for tick in _SPECTRUM_DBFS_TICKS:
        ty = bar_zone_y + bar_zone_h - int(bar_zone_h * _spectrum_y_ratio(tick))
        pygame.draw.line(surface, (70, 70, 70), (px + MARGIN, ty), (px + MARGIN + bar_area_w, ty), 1)
        label_key = ('spectrum_tick', tick)
        if label_key not in surf_cache:
            surf_cache[label_key] = font_scale.render(str(tick), True, (150, 150, 150))
        label = surf_cache[label_key]
        label_y = max(bar_zone_y, min(bar_zone_y + bar_zone_h - label.get_height(), ty - label.get_height() // 2))
        surface.blit(label, (px + MARGIN + 2, label_y))
        surface.blit(label, (px + pw - MARGIN - label.get_width() - 2, label_y))


_DIST_Y_LABELS = [(1.0, "100%"), (0.1, "10%"), (0.01, "1%"), (0.001, "0.1%"), (0.0001, "0.01%")]
_DIST_X_LABELS = [-50, -40, -30, -23, -20, -10]


def _dist_y_ratio(frac):
    """佔比 → 高度比例，對數刻度：100% 在頂、0.001% 在底，每個十倍一格（共 5 格）。"""
    if frac <= 0:
        return 0.0
    return min(1.0, max(0.0, float(np.log10(frac / _DIST_FRAC_LO) / -np.log10(_DIST_FRAC_LO))))


def draw_histogram_panel(surface, fonts, surf_cache, bars, title, px, pw, py, ph):
    """LUFS 分布長條圖，骨架同 draw_spectrum_panel。X 軸是 LUFS（每根 1 LU），每根用它代表的
    LUFS 值上色（lufs_to_color），-23 目標線畫成直線；Y 軸是佔比、對數刻度——響度分布集中在
    目標值附近，線性刻度會看不見兩側尾巴，而爆音與靜音段就藏在尾巴裡。"""
    font_title, font_scale = fonts[0], fonts[1]
    TITLE_H = 50
    MARGIN  = 4
    LABEL_H = font_scale.get_height() + 4      # 底部 X 軸刻度文字列

    title_key = ('dist_title', title)
    if title_key not in surf_cache:
        surf_cache[title_key] = font_title.render(title, True, (170, 170, 170))
    t = surf_cache[title_key]
    surface.blit(t, (px + pw // 2 - t.get_width() // 2, py + 8))

    n          = len(bars)
    bar_zone_y = py + TITLE_H
    bar_zone_h = ph - TITLE_H - MARGIN - LABEL_H
    bar_area_w = pw - MARGIN * 2
    bar_w      = max(2, (bar_area_w - MARGIN * (n - 1)) // n)

    pygame.draw.rect(surface, (28, 28, 28), (px + MARGIN, bar_zone_y, bar_area_w, bar_zone_h))

    total = int(bars.sum())
    if total > 0:
        for k in range(n):
            if bars[k] == 0:
                continue
            bh = max(2, int(bar_zone_h * _dist_y_ratio(bars[k] / total)))   # 有票的根至少 2px，尾巴才看得到
            bx = px + MARGIN + k * (bar_w + MARGIN)
            pygame.draw.rect(surface, lufs_to_color(_DIST_BAR_LO + k), (bx, bar_zone_y + bar_zone_h - bh, bar_w, bh))

    # Y 軸參考線（每個十倍一條）與兩側刻度文字
    for frac, text in _DIST_Y_LABELS:
        ty = bar_zone_y + bar_zone_h - int(bar_zone_h * _dist_y_ratio(frac))
        pygame.draw.line(surface, (70, 70, 70), (px + MARGIN, ty), (px + MARGIN + bar_area_w, ty), 1)
        label_key = ('dist_ytick', text)
        if label_key not in surf_cache:
            surf_cache[label_key] = font_scale.render(text, True, (150, 150, 150))
        label   = surf_cache[label_key]
        label_y = max(bar_zone_y, min(bar_zone_y + bar_zone_h - label.get_height(), ty - label.get_height() // 2))
        surface.blit(label, (px + MARGIN + 2, label_y))
        surface.blit(label, (px + pw - MARGIN - label.get_width() - 2, label_y))

    # -23 LUFS 目標線：判讀重點就是「分布有沒有騎在目標線上」
    tx = px + MARGIN + int(round(TARGET_LUFS - _DIST_BAR_LO)) * (bar_w + MARGIN) + bar_w // 2
    pygame.draw.line(surface, (255, 255, 255), (tx, bar_zone_y), (tx, bar_zone_y + bar_zone_h), 3)

    # X 軸刻度文字（LUFS），目標值用白色
    for v in _DIST_X_LABELS:
        is_target = (v == TARGET_LUFS)
        label_key = ('dist_xtick', v, is_target)
        if label_key not in surf_cache:
            surf_cache[label_key] = font_scale.render(str(v), True, (255, 255, 255) if is_target else (150, 150, 150))
        label = surf_cache[label_key]
        lx = px + MARGIN + (v - _DIST_BAR_LO) * (bar_w + MARGIN) + bar_w // 2 - label.get_width() // 2
        lx = max(px + MARGIN, min(lx, px + pw - MARGIN - label.get_width()))
        surface.blit(label, (lx, bar_zone_y + bar_zone_h + 2))


def draw_histogram_screen(surface, fonts, surf_cache, W, H, mode, hour_now):
    """Histogram 整個畫面：左側三格 LRA / P10 / P95，右側分布圖，版面同 Spectrum 畫面。"""
    left_w  = W // 5
    third_h = H // 3
    counts  = _dist_snapshot(mode)
    stats   = dist_lra(counts)
    if stats is None:
        lra_s = p10_s = p95_s = "---"
        p10_c = p95_c = (255, 255, 255)
    else:
        lra, p10, p95 = stats
        lra_s, p10_s, p95_s = f"{lra:.1f}", f"{round(p10)}", f"{round(p95)}"
        p10_c, p95_c = lufs_to_color(p10), lufs_to_color(p95)

    draw_number_only_panel(surface, fonts, surf_cache, "LRA", None, 0, left_w, 0, third_h,
                           val_str=lra_s, num_color=(255, 255, 255))
    draw_number_only_panel(surface, fonts, surf_cache, "P10", None, 0, left_w, third_h, third_h,
                           val_str=p10_s, num_color=p10_c)
    draw_number_only_panel(surface, fonts, surf_cache, "P95", None, 0, left_w, 2 * third_h, H - 2 * third_h,
                           val_str=p95_s, num_color=p95_c)
    for i in (1, 2):
        pygame.draw.line(surface, (55, 55, 55), (0, i * third_h), (left_w, i * third_h), 1)
    pygame.draw.line(surface, (55, 55, 55), (left_w, 0), (left_w, H), 1)

    title = f"LUFS HISTOGRAM - THIS HOUR ({hour_now})" if mode == "hour" else "LUFS HISTOGRAM - SLIDING (3')"
    draw_histogram_panel(surface, fonts, surf_cache, dist_bars(counts), title, left_w, W - left_w, 0, H)


_DISPLAY_CYCLE = ("loudness", "spectrum", "histogram")


def _toggle_display_mode():
    global display_mode
    display_mode = _DISPLAY_CYCLE[(_DISPLAY_CYCLE.index(display_mode) + 1) % len(_DISPLAY_CYCLE)]
    print(f"[GPIO DEBUG] toggle fired (GPIO{GPIO_PIN_TOGGLE}) -> display_mode={display_mode}", flush=True)


def _step_freq(delta):
    """上一個/下一個頻率鈕的共用處理——Spectrum 畫面步進頻率；Histogram 畫面則是兩顆都切換
    統計範圍（整點累積 ↔ 滑動 3 分鐘，只有兩個選項，方向沒有意義）；Loudness 畫面不生效，避免 PM5D 端誤觸。
    撞到邊界時繞到另一頭（wraparound），不是卡住不動：萬一其中一顆鍵故障，
    另一顆單獨連續按繞一圈還是能到達所有頻段，不會有走不到的死角。
    清單最後一格（BYPASS_INDEX）代表「不選定特定頻段」，純顯示狀態（BAND dBFS／
    BAND FREQ 顯示 ---／ALL），不影響任何音訊處理——Pi 本來就不輸出音訊。
    純粹更新 freq_index，不做任何運算，停在哪一格完全零成本。"""
    global freq_index, dist_mode
    pin = GPIO_PIN_FREQ_PREV if delta < 0 else GPIO_PIN_FREQ_NEXT
    if display_mode == "histogram":
        dist_mode = "slide" if dist_mode == "hour" else "hour"
        print(f"[GPIO DEBUG] step_freq fired (GPIO{pin}, delta={delta}) -> dist_mode={dist_mode}", flush=True)
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


def main():
    global current_hour, reset_hour_flag, _running, prev_H, prev_SEG, st_history, seg_history

    DEVICE = _setup_scarlett()

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

    # 以 1920 為基準做比例縮放，適應任何解析度
    _scale = W / 1920
    _fs_value  = max(80,  round(250 * _scale))
    _fs_lr     = max(50,  round(140 * _scale))
    _fs_delta  = max(32,  round(100 * _scale))   # delta 獨立字型，比 L/R 略小
    _fs_title  = max(24,  round(70  * _scale))
    _fs_scale  = max(16,  round(32  * _scale))

    font_title = pygame.font.SysFont("monospace", _fs_title, bold=True)
    font_scale = pygame.font.SysFont("monospace", _fs_scale)

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

    fonts = (font_title, font_scale, font_value, font_lr, font_delta)

    surf_cache = {}   # surface cache: keyed by (type, content, ...) — rebuilt only on change

    seg_snapshot_time = time.time() + 180.0   # 3 分鐘後第一次快照
    st_snapshot_time  = time.time() + 3.0     # 3 秒後第一次 SHORT TERM 快照

    pw = W // 3

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
            if now_t >= st_snapshot_time:
                with lock:
                    st_val = latest["S"]
                if st_val > -70:
                    st_history.append(st_val)
                st_snapshot_time = now_t + 3.0

            if now_t >= seg_snapshot_time:
                with lock:
                    seg_val = latest["SEG"]
                if seg_val > -70:
                    seg_history.append(seg_val)
                    prev_SEG = seg_val
                seg_snapshot_time = now_t + 180.0

            screen.fill((15, 15, 15))

            with lock:
                vals = dict(latest)

            if display_mode == "loudness":
                for idx, (title, val, lr, hist) in enumerate([
                    ("MOMENTARY",       vals["M"], (vals["M_L"], vals["M_R"]), None),
                    ('SHORT TERM (3")', vals["S"], None,                       st_history),
                ]):
                    draw_panel(screen, fonts, surf_cache, title, val, idx * pw, pw, H, lr_vals=lr, history=hist)

                # H panel: THIS HOUR (top) + SEGMENT (bottom), number only
                half_h    = H // 2
                h_delta   = round(vals["H"]) - round(prev_H)   if prev_H   is not None and vals["H"]   > -70 else None
                seg_delta = round(vals["SEG"]) - round(seg_history[-1]) if seg_history and vals["SEG"] > -70 else None
                draw_number_only_panel(screen, fonts, surf_cache, f"THIS HOUR ({current_hour})", vals["H"],   2 * pw, pw, 0,      half_h,       delta=h_delta, num_color=lufs_to_color(vals["H"]))
                pygame.draw.line(screen, (55, 55, 55), (2 * pw, half_h), (2 * pw + pw, half_h), 1)
                draw_number_only_panel(screen, fonts, surf_cache, "SEGMENT (3')",                vals["SEG"], 2 * pw, pw, half_h, H - half_h,   delta=seg_delta, history=seg_history, delta_align='side')

                for i in (1, 2):
                    pygame.draw.line(screen, (55, 55, 55), (i * pw, 0), (i * pw, H), 1)
            elif display_mode == "histogram":
                draw_histogram_screen(screen, fonts, surf_cache, W, H, dist_mode, current_hour)
            else:
                left_w      = W // 5
                band_levels = compute_spectrum_bands()
                _update_spectrum_peaks(band_levels)
                if freq_index == BYPASS_INDEX:
                    freq_str = "ALL"
                    dbfs_str = "---"
                else:
                    freq_hz  = OCTAVE_BANDS[freq_index][0]
                    freq_str = _fmt_freq(freq_hz)
                    dbfs_str = f"{round(_spectrum_peaks[freq_index])}"

                third_h = H // 3
                draw_number_only_panel(screen, fonts, surf_cache, "ALL LUFS", vals["M"], 0, left_w, 0, third_h,
                                        num_color=lufs_to_color(vals["M"]))
                draw_number_only_panel(screen, fonts, surf_cache, "BAND dBFS", None, 0, left_w, third_h, third_h,
                                        val_str=dbfs_str, num_color=(255, 255, 255))
                draw_number_only_panel(screen, fonts, surf_cache, "BAND FREQ", None, 0, left_w, 2 * third_h, H - 2 * third_h,
                                        val_str=freq_str, num_color=(255, 255, 255))
                for i in (1, 2):
                    pygame.draw.line(screen, (55, 55, 55), (0, i * third_h), (left_w, i * third_h), 1)
                pygame.draw.line(screen, (55, 55, 55), (left_w, 0), (left_w, H), 1)

                draw_spectrum_panel(screen, fonts, surf_cache, band_levels, left_w, W - left_w, 0, H,
                                     highlight_index=freq_index, peak_levels=_spectrum_peaks)

            pygame.display.flip()
            clock.tick(FPS)


if __name__ == "__main__":
    main()
