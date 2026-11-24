#!/bin/bash
# 開熱點前的唯讀檢查（不改任何設定）：bash check_wifi.sh
echo "== 網卡狀態";            nmcli -f DEVICE,TYPE,STATE,CONNECTION device
echo "== Wi-Fi 開關";          nmcli radio wifi
echo "== rfkill";              rfkill list wifi 2>/dev/null || echo "(rfkill 不可用)"
echo "== 國家代碼";            raspi-config nonint get_wifi_country 2>/dev/null || echo "(讀不到，可能需要 sudo)"
echo "== 監管區域";            iw reg get 2>/dev/null | head -3
echo "== 熱點需要的 dnsmasq";  dpkg -l dnsmasq-base 2>/dev/null | tail -1 || echo "(未安裝)"
echo "== 已存的連線";          nmcli -f NAME,TYPE,DEVICE,AUTOCONNECT connection show
echo "== eth0 位址";           ip -4 -o addr show eth0 | awk '{print $4}'
echo "== 近 7 天電壓不足次數"; journalctl -k --since "-7 days" 2>/dev/null | grep -ci undervoltage
