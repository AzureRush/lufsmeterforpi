import numpy as np
import sounddevice as sd
import pyloudnorm as pyln
import pygame
import sys
import threading
import collections
import datetime
from scipy.signal import lfilter

# ─── Config ───────────────────────────────────────────────────────────────────
DEVICE      = 2
SAMPLE_RATE = 48000
CHANNELS    = 2
BLOCK_SIZE  = 2400          # 50ms

MOMENTARY_WIN    = 0.4
SHORTTERM_WIN    = 3.0
TARGET_LUFS      = -23.0
YELLOW_THRESHOLD = -17.0
RED_THRESHOLD    = -14.0
FPS              = 20

# ─── K-weighting filter coefficients for 48kHz (ITU-R BS.1770-4) ─────────────
# Stage 1: High-shelf pre-filter
_PRE_B = np.array([1.53512485958697, -2.69169618940638,  1.19839281085285])
_PRE_A = np.array([1.0,              -1.69065929318241,  0.73248077421585])
# Stage 2: RLB high-pass
_RLB_B = np.array([1.0,  -2.0,              1.0             ])
_RLB_A = np.array([1.0,  -1.99004745483398, 0.99007225036621])

# ─── Audio state ──────────────────────────────────────────────────────────────
buf_momentary = collections.deque(maxlen=int(SAMPLE_RATE * MOMENTARY_WIN))
buf_shortterm = collections.deque(maxlen=int(SAMPLE_RATE * SHORTTERM_WIN))

# H: store per-block K-weighted energy [E_L, E_R]; ~1 MB for full hour
hour_block_energies = collections.deque(maxlen=72000)  # 1hr @ 50ms/block

# K-weighting filter states (shape (2,) per channel), persists across blocks
_zi_pre = [np.zeros(2) for _ in range(CHANNELS)]
_zi_rlb = [np.zeros(2) for _ in range(CHANNELS)]

meter_ms = pyln.Meter(SAMPLE_RATE)
latest = {"M": -70.0, "S": -70.0, "H": -70.0}
lock = threading.Lock()

current_hour = datetime.datetime.now().hour
reset_hour_flag = False


# ─── Audio processing ─────────────────────────────────────────────────────────
def lufs_safe(m, audio):
    try:
        v = m.integrated_loudness(audio)
        return v if np.isfinite(v) else -70.0
    except Exception:
        return -70.0


def k_weight_and_energy(audio):
    """Apply stateful K-weighting filter; return mean-square energy per channel."""
    global _zi_pre, _zi_rlb
    energies = []
    for ch in range(CHANNELS):
        sig = audio[:, ch]
        sig, _zi_pre[ch] = lfilter(_PRE_B, _PRE_A, sig, zi=_zi_pre[ch])
        sig, _zi_rlb[ch] = lfilter(_RLB_B, _RLB_A, sig, zi=_zi_rlb[ch])
        energies.append(float(np.mean(sig ** 2)))
    return energies


def compute_h(block_energies):
    """EBU R128 two-stage gated integrated loudness from block K-weighted energies."""
    if len(block_energies) < 2:
        return -70.0
    arr   = np.array(block_energies)            # (N, 2)
    power = np.maximum(np.sum(arr, axis=1), 1e-12)  # stereo: G_L = G_R = 1
    lufs  = -0.691 + 10.0 * np.log10(power)

    # Stage 1: absolute gate -70 LUFS
    m1 = lufs > -70.0
    if not np.any(m1):
        return -70.0

    # Stage 2: relative gate (ungated mean - 8 LU)
    rel = -0.691 + 10.0 * np.log10(np.mean(power[m1])) - 8.0
    m2  = lufs > rel
    if not np.any(m2):
        return -70.0

    return -0.691 + 10.0 * np.log10(np.mean(power[m2]))


def audio_callback(indata, frames, time_info, status):
    global _zi_pre, _zi_rlb, reset_hour_flag

    audio = indata.copy()

    if reset_hour_flag:
        hour_block_energies.clear()
        _zi_pre = [np.zeros(2) for _ in range(CHANNELS)]
        _zi_rlb = [np.zeros(2) for _ in range(CHANNELS)]
        reset_hour_flag = False

    buf_momentary.extend(audio)
    buf_shortterm.extend(audio)
    hour_block_energies.append(k_weight_and_energy(audio))

    arr_m = np.array(list(buf_momentary))
    arr_s = np.array(list(buf_shortterm))

    M = lufs_safe(meter_ms, arr_m) if len(arr_m) >= int(SAMPLE_RATE * MOMENTARY_WIN) else -70.0
    S = lufs_safe(meter_ms, arr_s) if len(arr_s) >= int(SAMPLE_RATE * SHORTTERM_WIN) else -70.0
    H = compute_h(list(hour_block_energies))

    with lock:
        latest["M"] = M
        latest["S"] = S
        latest["H"] = H


# ─── GUI ──────────────────────────────────────────────────────────────────────
def lufs_to_color(val):
    if val >= RED_THRESHOLD:
        return (220, 50, 50)
    elif val >= YELLOW_THRESHOLD:
        return (220, 200, 50)
    else:
        return (50, 200, 80)


_SCALE_MARKS = [0, -10, -20, -23, -30, -40, -50, -60]


def draw_panel(surface, fonts, title, val, px, pw, ph):
    """Draw one vertical meter panel."""
    font_title, font_scale, font_value = fonts

    TITLE_H  = 46
    SCALE_W  = 50          # left strip for dB scale
    PAD      = 6
    LO, HI   = -60.0, 0.0

    BAR_Y = TITLE_H
    BAR_X = px + SCALE_W + PAD
    BAR_W = pw - SCALE_W - PAD * 2
    BAR_H = ph - TITLE_H - PAD

    color = lufs_to_color(val)

    # Title
    t = font_title.render(title, True, (170, 170, 170))
    surface.blit(t, (px + pw // 2 - t.get_width() // 2, 8))

    # Bar background
    pygame.draw.rect(surface, (28, 28, 28), (BAR_X, BAR_Y, BAR_W, BAR_H))

    # Bar fill
    fill_h = int(BAR_H * (max(LO, min(HI, val)) - LO) / (HI - LO))
    pygame.draw.rect(surface, color, (BAR_X, BAR_Y + BAR_H - fill_h, BAR_W, fill_h))

    # dB scale on left strip
    for db in _SCALE_MARKS:
        ratio = (db - LO) / (HI - LO)
        y = BAR_Y + BAR_H - int(BAR_H * ratio)
        is_target = (db == -23)
        tick_color = (255, 255, 255) if is_target else (110, 110, 110)
        tick_len   = 10 if is_target else 6
        # tick line into bar
        pygame.draw.line(surface, tick_color,
                         (BAR_X - tick_len, y), (BAR_X, y), 1)
        # label, right-aligned against the tick
        label = str(db)
        ls = font_scale.render(label, True, tick_color)
        surface.blit(ls, (BAR_X - tick_len - ls.get_width() - 3,
                          y - ls.get_height() // 2))

    # Target line across bar
    ty = BAR_Y + BAR_H - int(BAR_H * (TARGET_LUFS - LO) / (HI - LO))
    pygame.draw.line(surface, (255, 255, 255), (BAR_X, ty), (BAR_X + BAR_W, ty), 2)

    # Big LUFS value, centered in bar area
    val_str = f"{val:.1f}" if val > -70 else "---"
    vs = font_value.render(val_str, True, color)
    vx = BAR_X + BAR_W // 2 - vs.get_width() // 2
    vy = BAR_Y + BAR_H // 2 - vs.get_height() // 2
    bg = pygame.Surface((vs.get_width() + 16, vs.get_height() + 10), pygame.SRCALPHA)
    bg.fill((15, 15, 15, 185))
    surface.blit(bg, (vx - 8, vy - 5))
    surface.blit(vs, (vx, vy))


def main():
    global current_hour, reset_hour_flag

    pygame.init()
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    W, H = screen.get_size()
    pygame.mouse.set_visible(False)
    clock = pygame.time.Clock()

    font_title = pygame.font.SysFont("monospace", 26, bold=True)
    font_scale = pygame.font.SysFont("monospace", 12)
    font_value = pygame.font.SysFont("monospace", 66, bold=True)
    fonts = (font_title, font_scale, font_value)

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
                    pygame.quit(); sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()

            # Hour reset check
            now_h = datetime.datetime.now().hour
            if now_h != current_hour:
                current_hour = now_h
                reset_hour_flag = True

            screen.fill((15, 15, 15))

            with lock:
                vals = dict(latest)

            for idx, (title, key) in enumerate([
                ("MOMENTARY",  "M"),
                ("SHORT TERM", "S"),
                ("THIS HOUR",  "H"),
            ]):
                draw_panel(screen, fonts, title, vals[key], idx * pw, pw, H)

            # Panel dividers
            for i in (1, 2):
                pygame.draw.line(screen, (55, 55, 55), (i * pw, 0), (i * pw, H), 1)

            pygame.display.flip()
            clock.tick(FPS)


if __name__ == "__main__":
    main()
