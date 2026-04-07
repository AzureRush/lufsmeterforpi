import numpy as np
import sounddevice as sd
import pyloudnorm as pyln
import pygame
import sys
import threading
import collections
import datetime
import time
from scipy.signal import lfilter

# ─── Config ───────────────────────────────────────────────────────────────────
DEVICE      = 2
SAMPLE_RATE = 48000
CHANNELS    = 2
BLOCK_SIZE  = 2400          # 50ms per block

MOMENTARY_WIN    = 0.4      # 400ms = 8 blocks
SHORTTERM_WIN    = 3.0      # 3s    = 60 blocks
TARGET_LUFS      = -23.0
YELLOW_THRESHOLD = -17.0
RED_THRESHOLD    = -14.0
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
hour_block_energies = collections.deque(maxlen=72000)  # 1hr K-weighted energies

# K-weighting filter states — must stay in audio callback to stay continuous
_zi_pre = [np.zeros(2) for _ in range(CHANNELS)]
_zi_rlb = [np.zeros(2) for _ in range(CHANNELS)]

meter_ms = pyln.Meter(SAMPLE_RATE)
latest   = {"M": -70.0, "S": -70.0, "H": -70.0, "M_L": -70.0, "M_R": -70.0}
lock     = threading.Lock()

current_hour    = datetime.datetime.now().hour
reset_hour_flag = False
_running        = True


# ─── Audio processing ─────────────────────────────────────────────────────────
def _energy_to_lufs(e):
    return -0.691 + 10.0 * np.log10(e)


def _fmt_lufs(v, prefix=""):
    return f"{prefix}{int(round(v))}" if v > -70 else f"{prefix}---"


def lufs_safe(m, audio):
    try:
        v = m.integrated_loudness(audio)
        return v if np.isfinite(v) else -70.0
    except Exception:
        return -70.0


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


def compute_loop():
    """Runs at ~10Hz in a separate thread. Does all heavy LUFS computation."""
    while _running:
        time.sleep(0.1)

        # Snapshot buffer state — fast list copies, GIL-protected
        m_snap = list(buf_m_chunks)
        s_snap = list(buf_s_chunks)
        h_snap = list(hour_block_energies)

        M = -70.0
        S = -70.0

        if len(m_snap) >= M_BLOCKS:
            m_audio = np.concatenate(m_snap)
            M = lufs_safe(meter_ms, m_audio)

            # Per-channel K-weighted level (non-stateful, fine for 400ms window)
            for ch_idx, key in enumerate(("M_L", "M_R")):
                sig = lfilter(_PRE_B, _PRE_A, m_audio[:, ch_idx])
                sig = lfilter(_RLB_B, _RLB_A, sig)
                e   = float(np.mean(sig ** 2))
                with lock:
                    latest[key] = _energy_to_lufs(e) if e > 1e-12 else -70.0

        if len(s_snap) >= S_BLOCKS:
            S = lufs_safe(meter_ms, np.concatenate(s_snap))

        H = compute_h(h_snap)

        with lock:
            latest["M"] = M
            latest["S"] = S
            latest["H"] = H


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
    else:
        return (50, 200, 80)


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


def draw_panel(surface, fonts, title, val, px, pw, ph, lr_vals=None):
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

    # Title
    surface.set_clip(pygame.Rect(px, 0, pw, TITLE_H))
    t = font_title.render(title, True, (170, 170, 170))
    surface.blit(t, (px + pw // 2 - t.get_width() // 2, 8))
    surface.set_clip(None)

    # Bar background
    pygame.draw.rect(surface, (28, 28, 28), (BAR_X, BAR_Y, BAR_W, BAR_H))

    # Bar fill
    fill_h = int(BAR_H * _y_ratio(val, LO, HI))
    pygame.draw.rect(surface, color, (BAR_X, BAR_Y + BAR_H - fill_h, BAR_W, fill_h))

    # dB scale
    for db in _SCALE_MARKS:
        y = BAR_Y + BAR_H - int(BAR_H * _y_ratio(db, LO, HI))
        is_target  = (db == -23)
        tick_color = (255, 255, 255) if is_target else (110, 110, 110)
        tick_len   = 10 if is_target else 6
        pygame.draw.line(surface, tick_color, (BAR_X - tick_len, y), (BAR_X, y), 1)
        ls = font_scale.render(str(db), True, tick_color)
        surface.blit(ls, (BAR_X - tick_len - ls.get_width() - 3, y - ls.get_height() // 2))

    # Target line
    ty = BAR_Y + BAR_H - int(BAR_H * _y_ratio(TARGET_LUFS, LO, HI))
    pygame.draw.line(surface, (255, 255, 255), (BAR_X, ty), (BAR_X + BAR_W, ty), 2)

    # Big value — auto-scale to panel width
    val_str = _fmt_lufs(val)
    vs_raw  = render_outlined(font_value, val_str, (255, 255, 255), offset=4)
    scale   = min((pw - MARGIN * 2) / vs_raw.get_width(),
                  (BAR_H - MARGIN * 2) / vs_raw.get_height())
    new_w   = int(vs_raw.get_width()  * scale)
    new_h   = int(vs_raw.get_height() * scale)
    vs      = pygame.transform.smoothscale(vs_raw, (new_w, new_h))
    vx      = px + pw // 2 - new_w // 2
    vy      = BAR_Y + BAR_H // 2 - new_h // 2
    surface.set_clip(pygame.Rect(px, BAR_Y, pw, BAR_H))
    surface.blit(vs, (vx, vy))
    surface.set_clip(None)

    # L/R per-channel text at bottom of bar (Momentary only)
    if lr_vals is not None:
        l_val, r_val = lr_vals
        l_str = _fmt_lufs(l_val, "L:")
        r_str = _fmt_lufs(r_val, "R:")
        l_surf = render_outlined(font_lr, l_str, (180, 180, 180), offset=2)
        r_surf = render_outlined(font_lr, r_str, (180, 180, 180), offset=2)
        gap    = 4
        total_h = l_surf.get_height() + gap + r_surf.get_height()
        ty_start = BAR_Y + BAR_H - total_h - 6
        surface.set_clip(pygame.Rect(BAR_X, BAR_Y, BAR_W, BAR_H))
        surface.blit(l_surf, (BAR_X + BAR_W // 2 - l_surf.get_width() // 2, ty_start))
        surface.blit(r_surf, (BAR_X + BAR_W // 2 - r_surf.get_width() // 2, ty_start + l_surf.get_height() + gap))
        surface.set_clip(None)


def main():
    global current_hour, reset_hour_flag, _running

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
                    pygame.quit(); sys.exit()

            now_h = datetime.datetime.now().hour
            if now_h != current_hour:
                current_hour = now_h
                reset_hour_flag = True

            screen.fill((15, 15, 15))

            with lock:
                vals = dict(latest)

            for idx, (title, val, lr) in enumerate([
                ("MOMENTARY",                   vals["M"], (vals["M_L"], vals["M_R"])),
                ("SHORT TERM (3s)",             vals["S"], None),
                (f"THIS HOUR ({current_hour})", vals["H"], None),
            ]):
                draw_panel(screen, fonts, title, val, idx * pw, pw, H, lr_vals=lr)

            for i in (1, 2):
                pygame.draw.line(screen, (55, 55, 55), (i * pw, 0), (i * pw, H), 1)

            pygame.display.flip()
            clock.tick(FPS)


if __name__ == "__main__":
    main()
