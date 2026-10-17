#!/bin/bash
# 啟動響度計（背景執行，log 寫到 /tmp/lm.log）。開機自動啟動與手動重啟都用這支。
# 換音訊介面（不是 Scarlett）：在 ~/.config/labwc/environment 加一行 LM_AUDIO_DEVICE=裝置名稱的一部分（例如 UMC）
# 參數 --autostart：開機自動啟動用，先等 3 秒讓 kanshi 套好解析度
# 參數 --restart：先停掉正在跑的再啟動（只比對 python 程序，SSH 遠端下也不會砍到自己的連線）
DIR="$(cd "$(dirname "$0")" && pwd)"
[ "${1:-}" = "--autostart" ] && sleep 3
if [ "${1:-}" = "--restart" ]; then
    pkill -f "python.*$DIR/loudness_meter.py" && sleep 2
fi

# 已經在跑就不重複開（開機自動啟動＋手動各跑一份，會搶同一個音訊介面）
if pgrep -f "python.*$DIR/loudness_meter.py" > /dev/null; then
    echo "already running"
    exit 0
fi

export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
export WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-wayland-0}"
export SDL_VIDEODRIVER=wayland
# stdin 接 /dev/null：無人值守啟動，找不到 Scarlett 時自動選裝置而不是停下來等鍵盤輸入
"$DIR/venv/bin/python" "$DIR/loudness_meter.py" > /tmp/lm.log 2>&1 < /dev/null &
echo "Started PID:$!"
