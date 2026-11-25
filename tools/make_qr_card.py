"""產生「手機遙控」QR 卡：① 掃描連上 Pi 熱點 ② 掃描打開遙控網頁。印出來貼在螢幕旁。

    py -3 tools/make_qr_card.py            # 會請你輸入熱點密碼（不顯示、不存檔），輸出 qr_card.png
    py -3 tools/make_qr_card.py out.png    # 指定輸出檔名

⚠ 輸出的圖含熱點密碼，已在 .gitignore（qr_card*.png），不要放進公開 repo。拿到卡的人都能連熱點、切畫面。
需要 segno、Pillow（Windows 的 Noto Sans TC 字型）。
"""
import getpass
import io
import os
import sys

import segno
from PIL import Image, ImageDraw, ImageFont

SSID = "MeterOnPi"
URL = "http://10.42.0.1:8080"
FONT = r"C:\Windows\Fonts\NotoSansTC-VF.ttf"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def wifi_payload(ssid, pw):
    # Wi-Fi QR 格式（Android 10+、iOS 11+ 相機都認得）：特殊字元 \ ; , : " 要加反斜線
    esc = lambda s: "".join("\\" + c if c in '\\;,:"' else c for c in s)
    return f"WIFI:T:WPA;S:{esc(ssid)};P:{esc(pw)};;"


def ask_password():
    while True:
        pw = getpass.getpass("熱點密碼（不顯示）：")
        if pw == getpass.getpass("再輸入一次：") and len(pw) >= 8:
            return pw
        print("兩次不一致或少於 8 個字元，請重新輸入")


def qr_image(data, px):
    # 每個 module 用整數倍像素，印小也不會一格粗一格細
    qr = segno.make(data, error="m")
    buf = io.BytesIO()
    qr.save(buf, kind="png", scale=px // qr.symbol_size(scale=1, border=2)[0], border=2)
    return Image.open(buf).convert("RGB")


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "qr_card.png")
    pw = ask_password()
    # 貼 Pi 盒子用：300 dpi 下 80×50 mm（以 100% 列印），QR 約 3 cm、字最小約 2.2 mm（≈ 7 pt）
    W, H, Q = 945, 591, 378
    card = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(card)
    f = lambda size, w=700: (lambda ft: (ft.set_variation_by_axes([w]), ft)[1])(ImageFont.truetype(FONT, size))
    for i, (data, title, note) in enumerate((
            (wifi_payload(SSID, pw), "① 連 Wi-Fi", SSID),
            (URL, "② 開遙控頁", URL.removeprefix("http://")))):
        cx = W // 4 + i * (W // 2)
        img = qr_image(data, Q)
        card.paste(img, (cx - img.width // 2, 70 + (Q - img.height) // 2))
        d.text((cx, 42), title, fill="black", font=f(42, 800), anchor="mm")
        d.text((cx, 70 + Q + 22), note, fill="black", font=f(30, 500), anchor="mm")
    d.text((W // 2, H - 34), "跳出「沒有網際網路」→ 選「保持連線」", fill=(60, 60, 60), font=f(28, 500), anchor="mm")
    del pw
    card.save(out, dpi=(300, 300))
    print(f"已輸出：{out}（含熱點密碼，勿公開）")


if __name__ == "__main__":
    main()
