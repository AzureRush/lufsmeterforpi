#!/usr/bin/env python3
"""響度計長時間監測：每 10 秒記一筆 CPU%、RSS、溫度、降頻旗標、log 行數，寫成 CSV。
用法：lm_monitor.py <響度計 PID> <輸出 csv> <結束 epoch 秒>
跑到結束時間、或響度計行程消失為止。只用標準函式庫，不吃響度計的資源。"""
import os
import subprocess
import sys
import time

pid, out, end = int(sys.argv[1]), sys.argv[2], float(sys.argv[3])
TICK = os.sysconf("SC_CLK_TCK")
INTERVAL = 10.0


def proc_ticks():
    with open(f"/proc/{pid}/stat") as f:
        s = f.read().rsplit(")", 1)[1].split()
    return int(s[11]) + int(s[12])      # utime + stime（全部執行緒合計）


def rss_kb():
    with open(f"/proc/{pid}/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    return -1


def vc(cmd):
    try:
        return subprocess.run(["vcgencmd", cmd], capture_output=True, text=True, timeout=3).stdout.strip().split("=")[1]
    except Exception:
        return "?"


def log_stats():
    """(log 行數, 累計溢位次數)。溢位次數 = 所有 `[AUDIO STATUS] ... ×N in Xs` 的 N 加總——
    每 10 秒一筆，事後相鄰兩筆相減就知道溢位落在哪 10 秒、當時在哪個畫面。"""
    try:
        lines = ovf = 0
        with open("/tmp/lm.log", encoding="utf-8", errors="replace") as f:
            for line in f:
                lines += 1
                if "[AUDIO STATUS]" in line and "×" in line:
                    ovf += int(line.split("×", 1)[1].split()[0])
        return lines, ovf
    except (OSError, ValueError):
        return -1, -1


with open(out, "w", buffering=1) as f:
    f.write("ts,epoch,cpu_pct,rss_kb,temp_c,throttled,load1,log_lines,ovf_total\n")
    prev_t, prev_ticks = time.time(), proc_ticks()
    next_t = prev_t + INTERVAL
    while time.time() < end:
        time.sleep(max(0.0, next_t - time.time()))
        next_t += INTERVAL
        try:
            now, ticks = time.time(), proc_ticks()
            cpu = (ticks - prev_ticks) / TICK / (now - prev_t) * 100
            prev_t, prev_ticks = now, ticks
            lines, ovf = log_stats()
            f.write(f"{time.strftime('%H:%M:%S')},{now:.0f},{cpu:.1f},{rss_kb()},{vc('measure_temp').rstrip(chr(39) + 'C')},"
                    f"{vc('get_throttled')},{os.getloadavg()[0]:.2f},{lines},{ovf}\n")
        except FileNotFoundError:            # 響度計行程不見了
            f.write(f"{time.strftime('%H:%M:%S')},{time.time():.0f},PROCESS_GONE,,,,,,\n")
            break
