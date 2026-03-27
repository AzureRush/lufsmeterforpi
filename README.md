# EBU R128 Loudness Meter

Real-time loudness meter on Raspberry Pi 4B, following EBU R128 / ITU-R BS.1770-4.

## Hardware

| Component | Detail |
|---|---|
| Raspberry Pi 4B | Raspberry Pi OS Bookworm 64-bit |
| Audio Interface | Focusrite Scarlett 2i2 3rd Gen (USB) |
| Display | HDMI monitor, 800x480 |

Signal chain: PC audio out -> Scarlett 2i2 -> Pi -> HDMI monitor

## Metrics

| Panel | Description |
|---|---|
| MOMENTARY | M - 400ms window |
| SHORT TERM (3s) | S - 3s window |
| THIS HOUR (N) | H - Integrated loudness since top of current hour, resets every hour |

Target: -23 LUFS (white line on each bar)

## Dependencies

```
pip3 install sounddevice pyloudnorm pygame scipy numpy
```

## Run

```bash
DISPLAY=:0 python3 ~/loudness_meter/loudness_meter.py
```

Press ESC to quit.

## Architecture

- Audio thread (sounddevice, 50ms blocks): K-weighting filter + buffer append only, ~0.34ms / 50ms budget
- Compute thread (10Hz): pyloudnorm M/S + gated H computation
- Display thread (pygame, 20fps): reads computed values, renders UI
