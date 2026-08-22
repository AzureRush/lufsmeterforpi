#!/bin/bash
XDG_RUNTIME_DIR=/run/user/1000 WAYLAND_DISPLAY=wayland-0 SDL_VIDEODRIVER=wayland \
  ~/loudness_meter/venv/bin/python ~/loudness_meter/loudness_meter.py > /tmp/lm.log 2>&1 &
echo "Started PID:$!"
