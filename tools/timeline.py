"""把 lm.log 的 [AUDIO STATUS] 行還原成時間軸。

log 沒有時間戳。每行「×N in Xs」= 過去 X 秒內共 N 次溢位（第一行的起點 = 行程啟動）。
最後一行的時間 = log 檔最後修改時間（--end），再用各行的 X 往回推每一行的時間區間，
把 N 依區間重疊比例平均分配到各分鐘（同 09-12 用的 /tmp/timeline.sh 的做法）。
用法：timeline.py <log> --end "YYYY-MM-DD HH:MM:SS" [--from "..."] [--to "..."]
"""
import argparse
import re
from collections import defaultdict
from datetime import datetime

ap = argparse.ArgumentParser()
ap.add_argument("log")
ap.add_argument("--end", required=True, help="log 最後一行的時間（= log 檔 mtime，Pi 時間）")
ap.add_argument("--from", dest="t0")
ap.add_argument("--to", dest="t1")
ap.add_argument("--hours", type=int, default=24, help="基準表列出最近幾個整點小時")
a = ap.parse_args()

fmt = "%Y-%m-%d %H:%M:%S"
ts = lambda s: datetime.strptime(s, fmt).timestamp()          # noqa: E731
t_end = ts(a.end)

ev = []
gpio = 0
for line in open(a.log, encoding="utf-8", errors="replace"):
    m = re.search(r"\[AUDIO STATUS\].*×(\d+) in (\d+)s", line)
    if m:
        ev.append((int(m.group(1)), int(m.group(2))))
    elif "[GPIO DEBUG]" in line:
        gpio += 1

if not ev:
    print("log 裡沒有任何 [AUDIO STATUS] 行（= 0 次溢位）；GPIO 事件行數:", gpio)
    raise SystemExit

# 往回推每行的時間區間
bins = defaultdict(float)          # 分鐘起點 epoch → 期望溢位次數
t = t_end
spans = []
for n, x in reversed(ev):
    spans.append((t - x, t, n))
    t -= x
spans.reverse()
print(f"log 事件行數 {len(ev)}，總溢位 {sum(n for n, _ in ev)}，GPIO 事件行 {gpio}")
print(f"還原出的行程起點: {datetime.fromtimestamp(spans[0][0]).strftime(fmt)}  （對照：應為該行程實際啟動時間）")

for s, e, n in spans:
    rate = n / (e - s)
    m = int(s // 60) * 60
    while m < e:
        ov = min(e, m + 60) - max(s, m)
        if ov > 0:
            bins[m] += rate * ov
        m += 60


def window_total(lo, hi):
    return sum(v for m, v in bins.items() if lo <= m < hi)


if a.t0 and a.t1:
    lo, hi = ts(a.t0), ts(a.t1)
    tot = window_total(lo, hi)
    secs = hi - lo
    print(f"\n視窗 {a.t0} ~ {a.t1}: 溢位 {tot:.0f} 次；平均 {tot / secs:.3f}/s；約損失 {tot / (secs * 20) * 100:.2f}% 的 block")
    print("每分鐘溢位率（次/秒），只列非零或每 5 分鐘一筆：")
    m = int(lo // 60) * 60
    while m < hi:
        r = bins.get(m, 0.0) / 60
        mm = datetime.fromtimestamp(m)
        if r > 0 or mm.minute % 5 == 0:
            print(f"  {mm.strftime('%H:%M')}  {r:6.3f}/s  {'#' * int(r * 40)}")
        m += 60
    peak = max(((bins.get(mm_, 0.0) / 60, mm_) for mm_ in range(int(lo // 60) * 60, int(hi), 60)), default=(0, 0))
    print(f"視窗內峰值: {peak[0]:.3f}/s @ {datetime.fromtimestamp(peak[1]).strftime('%H:%M')}")
else:
    last_hour = int(t_end // 3600) * 3600            # 最後一個「完整」小時 = 最後一個整點之前那一小時
    print(f"\n最近 {a.hours} 個完整整點小時（溢位次數 / 損失% / 峰值分鐘）：")
    prof = defaultdict(list)
    for k in range(a.hours, 0, -1):
        h0 = last_hour - k * 3600
        tot = window_total(h0, h0 + 3600)
        mins = [bins.get(h0 + i * 60, 0.0) / 60 for i in range(60)]
        pk = max(mins)
        for i, r in enumerate(mins):
            prof[i].append(r)
        print(f"  {datetime.fromtimestamp(h0).strftime('%m-%d %H:00')}  {tot:7.0f}  {tot / 72000 * 100:5.2f}%  峰值 {pk:5.2f}/s @ +{mins.index(pk):02d} 分")
    print("\n分鐘相位剖面（最近整點小時的平均溢位率，次/秒）：")
    for i in (0, 10, 20, 25, 30, 35, 40, 45, 50, 55, 58, 59):
        v = sum(prof[i]) / len(prof[i])
        print(f"  +{i:02d} 分  {v:6.3f}/s  {'#' * int(v * 40)}")
