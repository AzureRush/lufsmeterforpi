#!/bin/bash
# 把 Pi 內建 Wi-Fi（wlan0）開成熱點，給手機連線用。網路線（eth0）不受影響。
# 用法（要自己輸入 sudo 密碼）：sudo bash setup_hotspot.sh
# 依據：Raspberry Pi 官方教學 https://www.raspberrypi.com/tutorials/host-a-hotel-wifi-hotspot/
#       NetworkManager 手冊 https://networkmanager.dev/docs/api/latest/nmcli.html
# 熱點模式下 Pi 在 wlan0 的位址是 10.42.0.1，並會自動發 IP 給手機。重跑會重建熱點（可用來改密碼）。
# 關掉：sudo nmcli connection down meter-hotspot && sudo nmcli connection modify meter-hotspot connection.autoconnect no
# 移除：sudo nmcli connection delete meter-hotspot
set -euo pipefail
SSID="${HOTSPOT_SSID:-MeterOnPi}"
CON=meter-hotspot

[ "$(id -u)" = 0 ] || { echo "ERROR: 請用 sudo 執行：sudo bash $0" >&2; exit 1; }

echo "==> Wi-Fi 國家代碼（沒設定的話雙頻 Wi-Fi 會被停用）"
cc="$(raspi-config nonint get_wifi_country 2>/dev/null || true)"
if [ "$cc" != "TW" ]; then
    raspi-config nonint do_wifi_country TW
    echo "已設為 TW（原本：${cc:-未設定}）"
else
    echo "已是 TW"
fi
rfkill unblock wifi || true
nmcli radio wifi on

echo "==> 設定熱點密碼（至少 8 個字元，輸入時不會顯示）"
while :; do
    read -rsp "密碼：" PW; echo
    read -rsp "再輸入一次：" PW2; echo
    if [ "$PW" = "$PW2" ] && [ "${#PW}" -ge 8 ]; then break; fi
    echo "兩次不一致或少於 8 個字元，請重新輸入"
done

echo "==> 建立熱點 $SSID（2.4 GHz，手機相容性最好）"
nmcli connection delete "$CON" >/dev/null 2>&1 || true
nmcli device wifi hotspot ifname wlan0 con-name "$CON" ssid "$SSID" band bg password "$PW"
unset PW PW2
# 預設不會開機自動開，要另外設定（官方教學的做法）
nmcli connection modify "$CON" connection.autoconnect yes connection.autoconnect-priority 100

echo "==> 結果"
nmcli -f NAME,DEVICE,AUTOCONNECT,AUTOCONNECT-PRIORITY connection show --active
echo "wlan0: $(ip -4 -o addr show wlan0 | awk '{print $4}')   （手機連熱點後用這個位址）"
echo "eth0 : $(ip -4 -o addr show eth0 | awk '{print $4}')   （應該維持 192.168.10.1，筆電 SSH 用）"
