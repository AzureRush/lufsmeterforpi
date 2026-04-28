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
    """Render text with a solid outline by blitting 8 offset shadow copies."""
    shadow  = font.render(text, True, outline_color)
    base    = font.render(text, True, color)
    w = base.get_width()  + offset * 2
    h = base.get_height() + offset * 2
    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    for dx, dy in [(-offset, 0), (offset, 0), (0, -offset), (0, offset),
                   (-offset, -offset), (offset, -offset), (-offset, offset), (offset, offset)]:
        surf.blit(shadow, (dx + offset, dy + offset))
    surf.blit(base, (offset, offset))
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


def draw_panel(surface, fonts, surf_cache, title, val, px, pw, ph, lr_vals=None):
    font_title, font_scale, font_value, font_lr = fonts

    TITLE_H = 85
    SCALE_W = 80
    PAD     = 6
    MARGIN  = 10
    LO, HI  = -60.0, 0.0

    BAR_Y = TITLE_H
    BAR_X = px + SCALE_W + PAD
    BAR_W = pw - SCALE_W - PAD * 2
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

    # dB scale (labels cached — never change)
    for db in _SCALE_MARKS:
        y = BAR_Y + BAR_H - int(BAR_H * _y_ratio(db, LO, HI))
        is_target  = (db == -23)
        tick_color = (255, 255, 255) if is_target else (110, 110, 110)
        tick_len   = 10 if is_target else 6
        pygame.draw.line(surface, tick_color, (BAR_X - tick_len, y), (BAR_X, y), 1)
        scale_key = ('scale', db)
        if scale_key not in surf_cache:
            surf_cache[scale_key] = font_scale.render(str(db), True, tick_color)
        ls = surf_cache[scale_key]
        surface.blit(ls, (BAR_X - tick_len - ls.get_width() - 3, y - ls.get_height() // 2))

    # Target line
    ty = BAR_Y + BAR_H - int(BAR_H * _y_ratio(TARGET_LUFS, LO, HI))
    pygame.draw.line(surface, (255, 255, 255), (BAR_X, ty), (BAR_X + BAR_W, ty), 2)

    # Big value (cached — re-renders only when integer value changes)
    val_str = _fmt_lufs(val)
    num_key = ('num_bar', val_str, pw, BAR_H)
    if num_key not in surf_cache:
        vs_raw = render_outlined(font_value, val_str, (255, 255, 255), offset=4)
        sc     = min((pw - MARGIN * 2) / vs_raw.get_width(),
                     (BAR_H - MARGIN * 2) / vs_raw.get_height())
        surf_cache[num_key] = pygame.transform.smoothscale(
            vs_raw, (int(vs_raw.get_width() * sc), int(vs_raw.get_height() * sc)))
    vs           = surf_cache[num_key]
    new_w, new_h = vs.get_size()
    vx           = px + pw // 2 - new_w // 2
    vy           = BAR_Y + BAR_H // 2 - new_h // 2
    surface.set_clip(pygame.Rect(px, BAR_Y, pw, BAR_H))
    surface.blit(vs, (vx, vy))
    surface.set_clip(None)

    # L/R per-channel text at bottom of bar (Momentary only, cached by string)
    if lr_vals is not None:
        l_val, r_val = lr_vals
        l_str = _fmt_lufs(l_val, "L:")
        r_str = _fmt_lufs(r_val, "R:")
        for lr_str in (l_str, r_str):
            lr_key = ('lr', lr_str)
            if lr_key not in surf_cache:
                surf_cache[lr_key] = render_outlined(font_lr, lr_str, (180, 180, 180), offset=2)
        l_surf   = surf_cache[('lr', l_str)]
        r_surf   = surf_cache[('lr', r_str)]
        gap      = 4
        total_h  = l_surf.get_height() + gap + r_surf.get_height()
        ty_start = BAR_Y + BAR_H - total_h - 6
        surface.set_clip(pygame.Rect(BAR_X, BAR_Y, BAR_W, BAR_H))
        surface.blit(l_surf, (BAR_X + BAR_W // 2 - l_surf.get_width() // 2, ty_start))
        surface.blit(r_surf, (BAR_X + BAR_W // 2 - r_surf.get_width() // 2, ty_start + l_surf.get_height() + gap))
        surface.set_clip(None)


def draw_number_only_panel(surface, fonts, surf_cache, title, val, px, pw, py, ph, delta=None):
    font_title, _, font_value, font_lr = fonts

    TITLE_H  = 70
    DELTA_H  = 110   # 底部保留給 delta 小字（只有 delta 不為 None 時使用）
    MARGIN   = 10

    surface.set_clip(pygame.Rect(px, py, pw, ph))
    title_key = ('title', title)
    if title_key not in surf_cache:
        surf_cache[title_key] = font_title.render(title, True, (170, 170, 170))
    t = surf_cache[title_key]
    surface.blit(t, (px + pw // 2 - t.get_width() // 2, py + 8))
    surface.set_clip(None)

    num_h   = ph - TITLE_H - (DELTA_H if delta is not None else 0)
    val_str = _fmt_lufs(val)
    num_key = ('num_panel', val_str, pw, num_h)
    if num_key not in surf_cache:
        vs_raw = render_outlined(font_value, val_str, (255, 255, 255), offset=4)
        sc     = min((pw - MARGIN * 2) / vs_raw.get_width(),
                     (num_h - MARGIN * 2) / vs_raw.get_height())
        surf_cache[num_key] = pygame.transform.smoothscale(
            vs_raw, (int(vs_raw.get_width() * sc), int(vs_raw.get_height() * sc)))
    vs           = surf_cache[num_key]
    new_w, new_h = vs.get_size()
    vx           = px + pw // 2 - new_w // 2
    vy           = py + TITLE_H + num_h // 2 - new_h // 2
    surface.set_clip(pygame.Rect(px, py, pw, ph))
    surface.blit(vs, (vx, vy))
    surface.set_clip(None)

    # Delta vs 上一區段（可選）
    if delta is not None:
        d_str   = (f"+{delta}" if delta > 0 else str(delta))
        d_color = (220, 50, 50) if delta > 0 else ((110, 200, 110) if delta < 0 else (140, 140, 140))
        d_key   = ('delta', d_str, d_color)
        if d_key not in surf_cache:
            surf_cache[d_key] = render_outlined(font_lr, d_str, d_color, offset=2)
        ds   = surf_cache[d_key]
        d_zone_y = py + TITLE_H + num_h
        dy   = d_zone_y + (DELTA_H - ds.get_height()) // 2
        surface.set_clip(pygame.Rect(px, d_zone_y, pw, DELTA_H))
        surface.blit(ds, (px + pw // 2 - ds.get_width() // 2, dy))
        surface.set_clip(None)


def main():
    global current_hour, reset_hour_flag, _running, prev_H, prev_SEG

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
    font_value = pygame.font.SysFont("monospace", 250, bold=True)
    font_lr    = pygame.font.SysFont("monospace", 90, bold=True)
    fonts = (font_title, font_scale, font_value, font_lr)

    surf_cache = {}   # surface cache: keyed by (type, content, ...) — rebuilt only on change

    seg_snapshot_time = time.time() + 180.0   # 3 分鐘後第一次快照

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
            if now_t >= seg_snapshot_time:
                with lock:
                    seg_val = latest["SEG"]
                prev_SEG = seg_val if seg_val > -70 else None   # 每 3 分鐘快照
                seg_snapshot_time = now_t + 180.0

            screen.fill((15, 15, 15))

            with lock:
                vals = dict(latest)

            for idx, (title, val, lr) in enumerate([
                ("MOMENTARY",      vals["M"], (vals["M_L"], vals["M_R"])),
                ('SHORT TERM (3")', vals["S"], None),
            ]):
                draw_panel(screen, fonts, surf_cache, title, val, idx * pw, pw, H, lr_vals=lr)

            # H panel: THIS HOUR (top) + SEGMENT (bottom), number only
            half_h    = H // 2
            h_delta   = round(vals["H"]   - prev_H)   if prev_H   is not None and vals["H"]   > -70 else None
            seg_delta = round(vals["SEG"] - prev_SEG) if prev_SEG is not None and vals["SEG"] > -70 else None
            draw_number_only_panel(screen, fonts, surf_cache, f"THIS HOUR ({current_hour})", vals["H"],   2 * pw, pw, 0,      half_h,       delta=h_delta)
            pygame.draw.line(screen, (55, 55, 55), (2 * pw, half_h), (2 * pw + pw, half_h), 1)
            draw_number_only_panel(screen, fonts, surf_cache, "SEGMENT (3')",                vals["SEG"], 2 * pw, pw, half_h, H - half_h,   delta=seg_delta)

            for i in (1, 2):
                pygame.draw.line(screen, (55, 55, 55), (i * pw, 0), (i * pw, H), 1)

            pygame.display.flip()
            clock.tick(FPS)


if __name__ == "__main__":
    main()
