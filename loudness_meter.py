import numpy as np
import sounddevice as sd
import pyloudnorm as pyln
import pygame
import sys
import threading
import collections
import time

# --- Config ---
DEVICE = 2
SAMPLE_RATE = 48000
CHANNELS = 2
BLOCK_SIZE = 2400      # 50ms blocks

# EBU R128 windows
MOMENTARY_WIN  = 0.4
SHORTTERM_WIN  = 3.0

# Display
WIDTH, HEIGHT = 800, 480
FPS = 20

# LUFS thresholds
YELLOW_THRESHOLD = -17.0
RED_THRESHOLD    = -14.0
TARGET_LUFS      = -23.0
TP_ALERT_THRESH  = -1.0   # -1 dBTP alert

# --- Audio buffers ---
buf_momentary = collections.deque(maxlen=int(SAMPLE_RATE * MOMENTARY_WIN))
buf_shortterm = collections.deque(maxlen=int(SAMPLE_RATE * SHORTTERM_WIN))
buf_integrated = []
buf_lra_vals   = collections.deque(maxlen=3600)  # ~3min of S values @20Hz

meter = pyln.Meter(SAMPLE_RATE)

latest = {"M": -70.0, "S": -70.0, "I": -70.0, "TP": -70.0, "LRA": 0.0}
tp_alert_until = 0.0
lock = threading.Lock()

def lufs_safe(m, audio):
    try:
        val = m.integrated_loudness(audio)
        return val if np.isfinite(val) else -70.0
    except Exception:
        return -70.0

def compute_lra(s_values, integrated_lufs):
    if len(s_values) < 6:
        return 0.0
    arr = np.array(s_values)
    gated = arr[arr > -70.0]
    if integrated_lufs > -70.0:
        gated = gated[gated > (integrated_lufs - 20.0)]
    if len(gated) < 2:
        return 0.0
    low  = np.percentile(gated, 10)
    high = np.percentile(gated, 95)
    return max(0.0, high - low)

def audio_callback(indata, frames, time_info, status):
    global buf_integrated, tp_alert_until
    audio = indata.copy()
    buf_momentary.extend(audio)
    buf_shortterm.extend(audio)
    buf_integrated.append(audio)

    arr_m = np.array(list(buf_momentary))
    arr_s = np.array(list(buf_shortterm))

    M = lufs_safe(meter, arr_m) if len(arr_m) >= int(SAMPLE_RATE * MOMENTARY_WIN) else -70.0
    S = lufs_safe(meter, arr_s) if len(arr_s) >= int(SAMPLE_RATE * SHORTTERM_WIN) else -70.0

    if len(buf_integrated) >= 10:
        arr_i = np.concatenate(buf_integrated)
        I = lufs_safe(meter, arr_i)
    else:
        I = -70.0

    TP = 20 * np.log10(np.max(np.abs(audio)) + 1e-9)

    if S > -70.0:
        buf_lra_vals.append(S)

    LRA = compute_lra(list(buf_lra_vals), I)

    now = time.time()
    if TP > TP_ALERT_THRESH:
        tp_alert_until = now + 3.0

    with lock:
        latest["M"]   = M
        latest["S"]   = S
        latest["I"]   = I
        latest["TP"]  = TP
        latest["LRA"] = LRA

def lufs_to_color(val):
    if val >= RED_THRESHOLD:
        return (220, 50, 50)
    elif val >= YELLOW_THRESHOLD:
        return (220, 200, 50)
    else:
        return (50, 200, 80)

def lra_to_color(val):
    if val > 15.0:
        return (220, 50, 50)
    elif val > 10.0:
        return (220, 200, 50)
    else:
        return (50, 200, 80)

def draw_bar(surface, font, label, val, x, y, w, h,
             lo=-60.0, hi=0.0, target_val=None, color_fn=None):
    pygame.draw.rect(surface, (30, 30, 30), (x, y, w, h))
    clamp = max(lo, min(hi, val))
    fill_ratio = (clamp - lo) / (hi - lo)
    fill_h = int(h * fill_ratio)
    color = (color_fn(val) if color_fn else lufs_to_color(val))
    pygame.draw.rect(surface, color, (x, y + h - fill_h, w, fill_h))

    if target_val is not None:
        ty = y + h - int(h * ((target_val - lo) / (hi - lo)))
        pygame.draw.line(surface, (255, 255, 255), (x, ty), (x + w, ty), 1)

    lbl_s = font.render(label, True, (200, 200, 200))
    surface.blit(lbl_s, (x + w // 2 - lbl_s.get_width() // 2, y + h + 5))

    if label == "LRA":
        val_str = f"{val:.1f} LU" if val > 0 else "--"
    else:
        val_str = f"{val:.1f}" if val > -70 else "--"
    val_surf = font.render(val_str, True, color)
    surface.blit(val_surf, (x + w // 2 - val_surf.get_width() // 2, y - 22))

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("EBU R128 Loudness Meter")
    font     = pygame.font.SysFont("monospace", 16)
    font_big = pygame.font.SysFont("monospace", 20, bold=True)
    clock    = pygame.time.Clock()

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, device=DEVICE,
        channels=CHANNELS, dtype="float32",
        blocksize=BLOCK_SIZE, callback=audio_callback
    )

    with stream:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()

            screen.fill((15, 15, 15))

            with lock:
                vals = dict(latest)

            bar_w = 90
            gap   = 35
            total = 5 * bar_w + 4 * gap
            sx    = (WIDTH - total) // 2
            bar_h = 300
            bar_y = 80

            bars = [
                ("M",   vals["M"],   -60.0, 0.0,  TARGET_LUFS, lufs_to_color),
                ("S",   vals["S"],   -60.0, 0.0,  TARGET_LUFS, lufs_to_color),
                ("I",   vals["I"],   -60.0, 0.0,  TARGET_LUFS, lufs_to_color),
                ("TP",  vals["TP"],  -60.0, 0.0,  -1.0,        lufs_to_color),
                ("LRA", vals["LRA"],  0.0,  20.0, None,        lra_to_color),
            ]
            for i, (lbl, val, lo, hi, tgt, cfn) in enumerate(bars):
                bx = sx + i * (bar_w + gap)
                draw_bar(screen, font, lbl, val, bx, bar_y, bar_w, bar_h,
                         lo=lo, hi=hi, target_val=tgt, color_fn=cfn)

            title = font_big.render("EBU R128 Loudness Meter", True, (180, 180, 180))
            screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 15))

            tgt_lbl = font.render("Target: -23 LUFS  |  TP limit: -1 dBTP", True, (130, 130, 130))
            screen.blit(tgt_lbl, (WIDTH // 2 - tgt_lbl.get_width() // 2, HEIGHT - 22))

            if time.time() < tp_alert_until:
                alert_surf = pygame.Surface((WIDTH, 28), pygame.SRCALPHA)
                alert_surf.fill((200, 30, 30, 210))
                screen.blit(alert_surf, (0, 46))
                alert_text = font.render("! TRUE PEAK OVER  -1 dBTP !", True, (255, 255, 255))
                screen.blit(alert_text, (WIDTH // 2 - alert_text.get_width() // 2, 53))

            pygame.display.flip()
            clock.tick(FPS)

if __name__ == "__main__":
    main()
