import numpy as np
import sounddevice as sd
import pygame
import sys
import subprocess
import threading
import collections
import datetime
import time
import csv
import os
from scipy.signal import lfilter

# ─── Config ───────────────────────────────────────────────────────────────────
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "loudness_log.csv")
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

MOMENTARY_WIN    = 0.4      # 400ms = 8 blocks
SHORTTERM_WIN    = 3.0      # 3s    = 60 blocks
TARGET_LUFS      = -23.0
BLUE_THRESHOLD   = -30.0
YELLOW_THRESHOLD = -23.0
RED_THRESHOLD    = -10.0
FPS              = 20

M_BLOCKS = 8    # blocks needed for M window
S_BLOCKS = 60   # blocks needed for S window

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
hour_block_energies    = collections.deque(maxlen=72000)  # 1hr K-weighted energies
segment_block_energies = collections.deque(maxlen=3600)   # 3min sliding window

# K-weighting filter states — must stay in audio callback to stay continuous
_zi_pre = [np.zeros(2) for _ in range(CHANNELS)]
_zi_rlb = [np.zeros(2) for _ in range(CHANNELS)]

latest   = {"M": -70.0, "S": -70.0, "H": -70.0, "M_L": -70.0, "M_R": -70.0, "SEG": -70.0}
lock     = threading.Lock()

current_hour    = datetime.datetime.now().hour
reset_hour_flag = False
_running        = True

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


def compute_h(block_energies):
    """EBU R128 two-stage gated integrated loudness from stored block energies."""
    if len(block_energies) < 2:
        return -70.0
    arr   = np.array(block_energies)
    power = np.maximum(np.sum(arr, axis=1), 1e-12)
    lufs  = _energy_to_lufs(power)
    m1 = lufs > -70.0
    if not np.any(m1):
        return -70.0
    rel = _energy_to_lufs(np.mean(power[m1])) - 8.0
    m2  = lufs > rel
    if not np.any(m2):
        return -70.0
    return _energy_to_lufs(np.mean(power[m2]))


def audio_callback(indata, frames, time_info, status):
    """Lightweight: only K-weight + 3 deque appends. ~2ms per call."""
    global _zi_pre, _zi_rlb, reset_hour_flag
    audio = indata.copy()

    if reset_hour_flag:
        hour_block_energies.clear()
        _zi_pre = [np.zeros(2) for _ in range(CHANNELS)]
        _zi_rlb = [np.zeros(2) for _ in range(CHANNELS)]
        reset_hour_flag = False

    energies = k_weight_and_energy(audio)   # ~1.6ms, stateful — must stay here
    buf_m_chunks.append(audio)              # deque.append is GIL-atomic
    buf_s_chunks.append(audio)
    hour_block_energies.append(energies)
    segment_block_energies.append(energies)


def compute_loop():
    """Runs at ~10Hz in a separate thread. Does all heavy LUFS computation."""
    while _running:
        time.sleep(0.1)

        # Snapshot buffer state — fast list copies, GIL-protected
        m_snap   = list(buf_m_chunks)
        s_snap   = list(buf_s_chunks)
        h_snap   = list(hour_block_energies)
        seg_snap = list(segment_block_energies)

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

        H   = compute_h(h_snap)
        SEG = compute_h(seg_snap)

        with lock:
            latest["M"]   = M
            latest["S"]   = S
            latest["H"]   = H
            latest["SEG"] = SEG


# ─── Logging ──────────────────────────────────────────────────────────────────
def _log_hour(dt, lufs_val):
    """Append one row to the CSV log. Creates file with header if needed."""
    write_header = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["date", "hour", "lufs"])
        lufs_str = f"{lufs_val:.1f}" if lufs_val > -70 else "---"
        w.writerow([dt.strftime("%Y-%m-%d"), dt.strftime("%H"), lufs_str])


# ─── GUI ──────────────────────────────────────────────────────────────────────
def render_outlined(font, text, color, outline_color=(0, 0, 0), offset=2):
    """3 層渲染：黑色外框 → 同色描邊（加粗筆畫）→ 填充。"""
    inner = max(1, offset // 2)

    shadow_surf = font.render(text, True, outline_color)
    color_surf  = font.render(text, True, color)   # stroke 和 fill 共用同一張

    w = color_surf.get_width()  + offset * 2
    h = color_surf.get_height() + offset * 2
    surf = pygame.Surface((w, h), pygame.SRCALPHA)

    for dx, dy in [(-offset, 0), (offset, 0), (0, -offset), (0, offset),
                   (-offset, -offset), (offset, -offset), (-offset, offset), (offset, offset)]:
        surf.blit(shadow_surf, (dx + offset, dy + offset))

    for dx, dy in [(-inner, 0), (inner, 0), (0, -inner), (0, inner),
                   (-inner, -inner), (inner, -inner), (-inner, inner), (inner, inner)]:
        surf.blit(color_surf, (dx + offset, dy + offset))

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
    """Non-linear map: -60..-40 → bottom 20%, -40..0 → top 80%."""
    val = max(lo, min(hi, val))
    if val <= _SPLIT_DB:
        return _SPLIT_FRAC * (val - lo) / (_SPLIT_DB - lo)
    else:
        return _SPLIT_FRAC + (1.0 - _SPLIT_FRAC) * (val - _SPLIT_DB) / (hi - _SPLIT_DB)


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
    font_title, font_scale, font_value, font_lr = fonts

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
        vs_raw = render_outlined(font_value, val_str, (255, 255, 255), offset=8)
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
                surf_cache[lr_key] = render_outlined(font_lr, lr_str, (255, 255, 255), offset=4)
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


def draw_number_only_panel(surface, fonts, surf_cache, title, val, px, pw, py, ph, delta=None, history=None):
    font_title, _, font_value, font_lr = fonts

    TITLE_H        = 70
    OUTLINE_OFFSET = 8
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
    val_str = _fmt_lufs(val)
    num_key = ('num_panel', val_str, pw, num_h)
    if num_key not in surf_cache:
        vs_raw = render_outlined(font_value, val_str, (255, 255, 255), offset=OUTLINE_OFFSET)
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
            surf_cache[d_key] = render_outlined(font_lr, d_str, d_color, offset=4)
        ds = surf_cache[d_key]

        content_left   = vx + int(OUTLINE_OFFSET * sc)
        minus_w_sc     = int(font_value.size('-')[0] * sc)
        delta_x        = content_left + minus_w_sc // 2
        num_bottom     = vy + new_h
        delta_half_h   = ds.get_height() // 2
        delta_center_y = min(py + ph - delta_half_h - 6,
                             num_bottom + delta_half_h + 4)

        surface.blit(ds, (delta_x - ds.get_width() // 2, delta_center_y - delta_half_h))

    surface.set_clip(None)


def main():
    global current_hour, reset_hour_flag, _running, prev_H, prev_SEG, st_history, seg_history

    DEVICE = _setup_scarlett()

    # Start compute thread before opening audio stream
    ct = threading.Thread(target=compute_loop, daemon=True)
    ct.start()

    pygame.init()
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    W, H   = screen.get_size()
    pygame.mouse.set_visible(False)
    clock  = pygame.time.Clock()

    font_title = pygame.font.SysFont("monospace", 70, bold=True)
    font_scale = pygame.font.SysFont("monospace", 32)

    dseg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DSEG7Modern-Bold.ttf")
    try:
        font_value = pygame.font.Font(dseg_path, 250)
        font_lr    = pygame.font.Font(dseg_path, 140)
        print(f"DSEG7Modern loaded: {dseg_path}")
    except (FileNotFoundError, pygame.error) as e:
        print(f"DSEG7Modern not found ({e}), fallback to monospace")
        font_value = pygame.font.SysFont("monospace", 250, bold=True)
        font_lr    = pygame.font.SysFont("monospace", 140, bold=True)

    fonts = (font_title, font_scale, font_value, font_lr)

    surf_cache = {}   # surface cache: keyed by (type, content, ...) — rebuilt only on change

    seg_snapshot_time = time.time() + 180.0   # 3 分鐘後第一次快照
    st_snapshot_time  = time.time() + 3.0     # 3 秒後第一次 SHORT TERM 快照

    pw = W // 3

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
                    with lock:
                        h_val = latest["H"]
                    _log_hour(datetime.datetime.now(), h_val)
                    pygame.quit(); sys.exit()

            now_h = datetime.datetime.now().hour
            if now_h != current_hour:
                with lock:
                    h_val = latest["H"]
                _log_hour(datetime.datetime.now().replace(hour=current_hour, minute=0, second=0), h_val)
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

            for idx, (title, val, lr, hist) in enumerate([
                ("MOMENTARY",       vals["M"], (vals["M_L"], vals["M_R"]), None),
                ('SHORT TERM (3")', vals["S"], None,                       st_history),
            ]):
                draw_panel(screen, fonts, surf_cache, title, val, idx * pw, pw, H, lr_vals=lr, history=hist)

            # H panel: THIS HOUR (top) + SEGMENT (bottom), number only
            half_h    = H // 2
            h_delta   = round(vals["H"]   - prev_H)       if prev_H   is not None and vals["H"]   > -70 else None
            seg_delta = round(vals["SEG"] - seg_history[-1]) if seg_history and vals["SEG"] > -70 else None
            draw_number_only_panel(screen, fonts, surf_cache, f"THIS HOUR ({current_hour})", vals["H"],   2 * pw, pw, 0,      half_h,       delta=h_delta)
            pygame.draw.line(screen, (55, 55, 55), (2 * pw, half_h), (2 * pw + pw, half_h), 1)
            draw_number_only_panel(screen, fonts, surf_cache, "SEGMENT (3')",                vals["SEG"], 2 * pw, pw, half_h, H - half_h,   delta=seg_delta, history=seg_history)

            for i in (1, 2):
                pygame.draw.line(screen, (55, 55, 55), (i * pw, 0), (i * pw, H), 1)

            pygame.display.flip()
            clock.tick(FPS)


if __name__ == "__main__":
    main()
