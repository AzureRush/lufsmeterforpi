#!/bin/bash
# 一鍵安裝響度計：在乾淨的 Raspberry Pi OS 桌面版（64-bit）上
#   git clone https://github.com/AzureRush/lufsmeterforpi.git ~/loudness_meter
#   cd ~/loudness_meter && bash install.sh
# 完成後重開機，登入桌面時會自動啟動。重複執行安全（已裝的會跳過或覆蓋）。需要網路（apt / pip）。
#
# 可調：LM_RESOLUTION=1920x1080@60Hz bash install.sh   （預設 1280x720@60Hz）
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
RES="${LM_RESOLUTION:-1280x720@60Hz}"
step() { echo; echo "==> $*"; }
fail() { echo "ERROR: $*" >&2; exit 1; }

step "檢查環境"
[ "$(id -u)" != 0 ]          || fail "請用一般使用者執行（不要加 sudo），需要時腳本會自己呼叫 sudo"
[ "$(uname -m)" = aarch64 ]  || fail "需要 64-bit Raspberry Pi OS（目前是 $(uname -m)）——Rust 模組只編了 64-bit 版"
command -v labwc >/dev/null  || fail "找不到 labwc 桌面：請用 Raspberry Pi OS 桌面版（Desktop），Lite 版沒有桌面"
python3 -c 'import sys; sys.exit(sys.version_info < (3, 8))' || fail "需要 Python 3.8 以上"
ls "$DIR"/wheels/lufs_core-*.whl >/dev/null 2>&1 || fail "找不到 wheels/lufs_core-*.whl（Rust 模組安裝檔）"

step "安裝系統套件（apt）"
sudo apt-get update
sudo apt-get install -y python3-venv python3-gpiozero python3-lgpio libportaudio2 kanshi wlr-randr grim

step "使用者群組（GPIO、音訊）"
for g in gpio audio; do
    if id -nG | grep -qw "$g"; then echo "已在 $g 群組"; else sudo usermod -aG "$g" "$USER"; echo "已加入 $g（重開機後生效）"; fi
done

step "Python 環境（venv，共用系統的 gpiozero / lgpio）"
[ -x "$DIR/venv/bin/python" ] || python3 -m venv --system-site-packages "$DIR/venv"
"$DIR/venv/bin/pip" install --upgrade pip
"$DIR/venv/bin/pip" install -r "$DIR/requirements.txt"
"$DIR/venv/bin/pip" install --force-reinstall --no-deps "$DIR"/wheels/lufs_core-*.whl

step "自我檢查"
"$DIR/venv/bin/python" -c "import numpy, scipy, sounddevice, pygame, gpiozero, lufs_core; lufs_core.EnergyStore(10); print('OK：所有套件都載入成功')"
chmod +x "$DIR/run.sh"

step "螢幕解析度（kanshi，Pi OS 桌面開機時會自動套用；不改 /boot 的開機檔）"
KANSHI=~/.config/kanshi/config
mkdir -p ~/.config/kanshi
touch "$KANSHI"
if [ -s "$KANSHI" ] && ! grep -q "loudness_meter" "$KANSHI"; then
    echo "~/.config/kanshi/config 已有其他設定，不覆蓋。需要的話手動加：output HDMI-A-1 mode $RES"
else
    # 兩個 HDMI 口各一個 profile：kanshi 依「目前接著的螢幕」挑符合的那個
    cat > "$KANSHI" <<EOF
# loudness_meter：由 install.sh 寫入（重跑會覆蓋這份）
profile hdmi1 {
    output HDMI-A-1 mode $RES
}
profile hdmi2 {
    output HDMI-A-2 mode $RES
}
EOF
    echo "已設定 $RES"
fi

step "開機自動啟動（~/.config/autostart，桌面登入後執行；不影響工作列與桌面）"
mkdir -p ~/.config/autostart
cat > ~/.config/autostart/loudness_meter.desktop <<EOF
[Desktop Entry]
Type=Application
Name=Loudness Meter
Exec=/bin/bash "$DIR/run.sh" --autostart
X-GNOME-Autostart-enabled=true
EOF
echo "已設定"

step "完成"
cat <<EOF
- 重開機（sudo reboot）後會自動進桌面並啟動響度計
- 手動重啟：bash "$DIR/run.sh" --restart
- 執行紀錄：/tmp/lm.log
- 換音訊介面（不是 Scarlett）：在 ~/.config/labwc/environment 加一行 LM_AUDIO_DEVICE=裝置名稱的一部分（例如 UMC），重開機生效
EOF
