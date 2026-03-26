import numpy as np
import sounddevice as sd
import pyloudnorm as pyln
import pygame
import sys
import threading
import collections
import time

# --- Config ---
DEVICE = 2          # None = default (Scarlett 2i2 if set as default), or use index
SAMPLE_RATE = 48000
CHANNELS = 2
BLOCK_SIZE = 2400      # 50ms blocks

# EBU R128 windows (in seconds)
MOMENTARY_WIN  = 0.4   # 400ms
SHORTTERM_WIN  = 3.0   # 3s

# Display
WIDTH, HEIGHT = 800, 480
FPS = 20

# LUFS thresholds for color
YELLOW_THRESHOLD = -23.0 + 6   # -17 LUFS
RED_THRESHOLD    = -23.0 + 9   # -14 LUFS
TARGET_LUFS      = -23.0

# --- Audio buffer ---
buf_momentary = collections.deque(maxlen=int(SAMPLE_RATE * MOMENTARY_WIN))
buf_shortterm = collections.deque(maxlen=int(SAMPLE_RATE * SHORTTERM_WIN))
buf_integrated = []

meter = pyln.Meter(SAMPLE_RATE)

latest = {"M": -70.0, "S": -70.0, "I": -70.0, "TP": -70.0}
lock = threading.Lock()

def lufs_safe(meter, audio):
    try:
        val = meter.integrated_loudness(audio)
        return val if np.isfinite(val) else -70.0
    except Exception:
        return -70.0

def audio_callback(indata, frames, time_info, status):
    global buf_integrated
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

    # True Peak (simple max abs)
    TP = 20 * np.log10(np.max(np.abs(audio)) + 1e-9)

    with lock:
        latest["M"] = M
        latest["S"] = S
        latest["I"] = I
        latest["TP"] = TP

# --- GUI ---
def lufs_to_color(val):
    if val >= RED_THRESHOLD:
        return (220, 50, 50)
    elif val >= YELLOW_THRESHOLD:
        return (220, 200, 50)
    else:
        return (50, 200, 80)

def draw_bar(surface, font, label, val, x, y, w, h):
    # Background
    pygame.draw.rect(surface, (30, 30, 30), (x, y, w, h))
    # Bar: map -60..0 LUFS to height
    clamp = max(-60.0, min(0.0, val))
    fill_ratio = (clamp + 60.0) / 60.0
    fill_h = int(h * fill_ratio)
    color = lufs_to_color(val)
    pygame.draw.rect(surface, color, (x, y + h - fill_h, w, fill_h))
    # Target line at -23 LUFS
    target_y = y + h - int(h * ((-23.0 + 60.0) / 60.0))
    pygame.draw.line(surface, (255, 255, 255), (x, target_y), (x + w, target_y), 1)
    # Label
    lbl = font.render(label, True, (200, 200, 200))
    surface.blit(lbl, (x + w // 2 - lbl.get_width() // 2, y + h + 5))
    # Value
    val_str = f"{val:.1f}" if val > -70 else "--"
    val_surf = font.render(val_str, True, color)
    surface.blit(val_surf, (x + w // 2 - val_surf.get_width() // 2, y - 22))

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("EBU R128 Loudness Meter")
    font = pygame.font.SysFont("monospace", 18)
    font_big = pygame.font.SysFont("monospace", 22, bold=True)
    clock = pygame.time.Clock()

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        device=DEVICE,
        channels=CHANNELS,
        dtype="float32",
        blocksize=BLOCK_SIZE,
        callback=audio_callback
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

            # Draw bars
            bar_w = 100
            gap = 60
            total = 4 * bar_w + 3 * gap
            start_x = (WIDTH - total) // 2
            bar_h = 300
            bar_y = 80

            labels = [("M", vals["M"]), ("S", vals["S"]), ("I", vals["I"]), ("TP", vals["TP"])]
            for i, (lbl, val) in enumerate(labels):
                bx = start_x + i * (bar_w + gap)
                draw_bar(screen, font, lbl, val, bx, bar_y, bar_w, bar_h)

            # Title
            title = font_big.render("EBU R128 Loudness Meter", True, (180, 180, 180))
            screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 15))

            # Target label
            target_lbl = font.render("Target: -23 LUFS", True, (150, 150, 150))
            screen.blit(target_lbl, (WIDTH // 2 - target_lbl.get_width() // 2, HEIGHT - 25))

            pygame.display.flip()
            clock.tick(FPS)

if __name__ == "__main__":
    main()
