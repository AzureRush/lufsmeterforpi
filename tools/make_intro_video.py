"""介紹短片：四個畫面的預覽 GIF ＋ 框選 ＋ 中文字卡 → MP4（社群分享用）。

用法：
  python tools/make_intro_video.py                 # 輸出 intro.mp4
  python tools/make_intro_video.py --still 5 14.5  # 只輸出指定秒數的樣張 PNG
需要 Pillow 與 PATH 上的 ffmpeg。素材是 assets/*_preview.gif（1280x720）；
PHASE 兩段是 intro_src/phase/*.png，要先在有 pygame 的環境跑 tools/render_phase_frames.py 產生；
遙控卡的手機截圖 intro_src/remote_phone.png 是遙控頁在 PHASE 畫面時的截圖（Chrome headless 500×1000、2 倍）。
⚠ 專案根目錄有 qr_card.png（tools/make_qr_card.py 產生）時，遙控卡會直接放整張 QR 卡——**含熱點密碼**，
  輸出的影片只能在組內流通（使用者 2026-11-26 決定）。沒有 qr_card.png 時改回文字提示。
"""
import functools
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageSequence

ROOT = Path(__file__).resolve().parent.parent
W, H, FPS = 1920, 1080, 30
GX, GY, GW, GH = 192, 20, 1536, 864        # GIF 在畫布上的位置（1280x720 → 1.2 倍）
SC = GW / 1280
BG = (13, 17, 23)
ACCENT = (88, 166, 255)
FONT = r"C:\Windows\Fonts\NotoSansTC-VF.ttf"
FADE = 0.35        # 框選區切換的交叉淡化秒數
DIM = 150          # 框外調暗程度（0~255）

COLOR_LEGEND = [((50, 130, 220), "−30 以下"), ((50, 200, 80), "−30 ~ −23"),
                ((220, 200, 50), "−23 ~ −10"), ((220, 50, 50), "−10 以上")]
COLOR_NOTE = "顏色分區為本系統自訂；EBU R128 的 −23 是整段節目的平均目標，瞬間值上下跳動屬正常"

SELECTED = "selected"   # 框選區代號：Spectrum 目前選定的那根長條（逐幀偵測位置）

# 段落：(GIF, 長度秒, 使用的 GIF 幀範圍, 標籤, 按鍵說明, [(開始秒, [框選區...], 標題, 說明)])
# 框選區是 GIF 原始 1280x720 座標
SEGMENTS = [
    ("card", 6.0, "about_loudness"),
    ("loudness", 14.0, (0, 40), "畫面 1　LOUDNESS", "SUB1 only　BANK A　KEY 9 下一個畫面　KEY 10／11 不作用", [
        (0.0, [(0, 0, 426, 720)], "MOMENTARY｜這一瞬間有多大聲", "約 0.4 秒的平均，反應最快；下方 L / R 是左右聲道"),
        (3.3, [(427, 0, 851, 720)], "SHORT TERM｜最近 3 秒的平均", "比 MOMENTARY 穩定，適合看「這一段」整體多大聲"),
        (6.6, [(0, 368, 851, 720)], "白線＝目標 −23 LUFS　顏色＝數值落在哪一區", "legend"),
        (10.6, [(852, 0, 1280, 720)], "THIS HOUR｜這一小時　SEGMENT｜最近 3 分鐘",
         "這兩個才是對照 −23 目標的數字；小字＝和上一小時／3 分鐘前差幾 LU"),
    ]),
    ("card", 6.0, "about_spectrum"),
    ("spectrum", 14.0, (0, 40), "畫面 2　SPECTRUM", "SUB1 only　BANK A　KEY 9 下一個畫面　KEY 10 往低頻　KEY 11 往高頻", [
        (0.0, [(258, 50, 1280, 720)], "每根長條＝一個頻段的電平，越高電平越大",
         "25 Hz ~ 20 kHz 共 30 段；刻度單位 dBFS，−20 ~ −40 放大顯示"),
        (3.5, [(258, 50, 1280, 720)], "細白線＝剛剛衝到的最高點", "停留 3 秒後慢慢下滑（Peak hold）"),
        (7.0, [(0, 240, 257, 720), SELECTED], "藍色長條＋黃線＝目前選的頻段",
         "BAND dBFS＝此頻段的電平最高點（黃線）　BAND FREQ＝中心頻率（4＝4 kHz）"),
        (10.5, [(0, 0, 257, 240)], "ALL LUFS＝整體現在多大聲", "和畫面 1 的 MOMENTARY 是同一個數字"),
    ]),
    ("card", 6.0, "about_histogram"),
    # GIF 第 16 幀由 THIS HOUR 切到 SLIDING；幀範圍 8~44 配 14 秒 → 切換落在 3.12 秒（再早一點會停在第 15 幀）
    ("histogram", 14.0, (8, 44), "畫面 3　HISTOGRAM", "SUB1 only　BANK A　KEY 9 下一個畫面　KEY 10／11 切換範圍", [
        (0.0, [(258, 0, 1280, 720)], "響度歷史圖｜橫軸＝時間　縱軸＝響度（LUFS）",
         "THIS HOUR：整點起 0 ~ 60 分，每根約 7 秒，從左往右長滿，整點後清空重來"),
        (3.12, [(258, 0, 1280, 50), (258, 668, 1280, 720)], "按 KEY 10／11 → SLIDING｜最近 3 分鐘",
         "每根 3 秒、共 60 根，每 3 秒往左捲一格；左舊右新"),
        (6.8, [(258, 50, 1280, 720)], "每根高度＝那段時間 SHORT TERM 的最高值",
         "顏色規則同畫面 1；白線＝−23；刻度與畫面 2 對齊"),
        (10.4, [(0, 480, 257, 720)], "LRA｜這段時間音量起伏多大", "數字越小越平穩，越大忽大忽小的落差越大（單位 LU）"),
    ]),
    ("card", 6.0, "about_phase"),
    ("card", 10.0, "phase_card"),
    # PHASE 兩段：render_phase_frames.py 用測試訊號渲染，20 FPS 即時速度（幀數＝秒數×20）。訊號切換秒數寫在 CLIPS
    ("phase_gonio", 15.0, (0, 300), "畫面 4　PHASE｜GONIO 頁", "SUB1 only　BANK A　KEY 9 回到 Loudness　KEY 10／11 切換 GONIO／BAND", [
        (0.0, [(0, 0, 256, 240), (256, 658, 1280, 716)], "CORR｜左右兩支麥克風的訊號像不像",
         "+1＝完全一樣　0＝毫無關係　−1＝極性相反；下方指針條是同一個數字"),
        (4.0, [(472, 58, 1064, 650)], "GONIOMETER｜把左右訊號畫成圖形",
         "直立的線＝同相；橫躺的線＝極性相反（例如線材 Pin 2／3 接反）"),
        (8.5, [(0, 240, 256, 480)], 'MIN (3")｜最近 3 秒內 CORR 的最低值',
         "反相只閃一下也留得住；兩人輪流說話時偶爾出現紅色負數屬正常"),
        (11.8, [(0, 480, 256, 720), (472, 58, 1064, 650)], "BAL dB｜左右音量差幾 dB",
         "L 10＝左邊大 10 dB，圖形也往 L 那側傾；只有一邊有聲音時只顯示 L 或 R"),
    ]),
    ("phase_band", 14.0, (0, 280), "畫面 4　PHASE｜BAND 頁", "SUB1 only　BANK A　KEY 9 回到 Loudness　KEY 10／11 切換 GONIO／BAND", [
        (0.0, [(256, 50, 1280, 650)], "BAND CORRELATION｜每個頻段各自的 CORR",
         "綠＝同相、紅＝反相；一段綠一段紅＝同一個聲音進了兩支麥克風、但有時間差（梳狀濾波）"),
        (3.5, [(0, 240, 256, 480), (256, 50, 1280, 650)], "NOTCH dB｜兩支一起推時，被吃掉最多的頻段少幾 dB",
         "黃框＝那個頻段（低於 −3 dB 才框）；越負聽起來越悶、越空"),
        (7.3, [(0, 240, 256, 720)], "BAL 決定凹洞能多深",
         "左右一樣大聲（BAL 0）凹到 −17；差 10 dB（L 10）只剩 −3——長條一樣紅綠交錯，悶不悶看 NOTCH"),
        (10.8, [(0, 0, 256, 240), (256, 658, 1280, 716)], "兩頁都有 CORR 與指針條",
         "梳狀濾波時整體 CORR 不會變負，所以要看這一頁的分頻長條與 NOTCH"),
    ]),
]
TITLE = "樹莓派多功能響度計"
PHOTO1, PHOTO2 = ROOT / "intro_src" / "photo1.jpg", ROOT / "intro_src" / "photo2.jpg"
# photo1（LM6505 現場照，原圖 1474x1110）要打馬賽克的區域：上排監看、左邊節目畫面、資產標籤
PHOTO1_MOSAIC = [(0, 0, 1474, 95), (70, 270, 630, 608), (770, 668, 1056, 768)]
# photo2（PM5D 面板，原圖 1110x1474）要框的按鍵：(框, 標籤)
MK_STEREO_A = ((415, 125, 495, 215), "STEREO A")
MK_DEFINE = ((415, 315, 495, 405), "DEFINE")
MK_LEVEL = ((95, 400, 240, 510), "")      # 面板上已印 LEVEL／ON，標籤會壓到旁邊按鍵，只框不標
MK_ON = ((135, 570, 205, 640), "")
MK_KEYS = ((130, 1205, 390, 1300), "KEY 9／10／11（SUB1 only）")
MK_KEY12 = ((410, 1205, 500, 1300), "KEY 12")
SIGNAL_CARD = ([MK_STEREO_A, MK_DEFINE, MK_LEVEL, MK_KEYS], [
    ("訊號從哪來", "h"),
    ("PM5D MONITOR OUT → 錄音介面 → 樹莓派", "chain"),
    ("畫面量的是 PM5D「MONITOR 目前選的訊號」，", "p"),
    ("換監聽來源，畫面上的數值就跟著換。", "p"),
    ("平常：MONITOR 選 STEREO A → 量主輸出", "li"),
    ("要量某個音軌：把它送進 MIX 20，", "li"),
    ("關掉 STEREO A、打開 DEFINE（兩個都開會疊加）", "li2"),
    ("MONITOR LEVEL 旋鈕位置已校正，轉動會改變讀數", "warn"),
])
PHASE_CARD = [
    ("PHASE 畫面的前提", "h"),
    ("兩支麥克風要分開送進左、右聲道", "chain"),
    ("畫面比的是「左聲道 vs 右聲道」。直接看節目 L/R 時，", "p"),
    ("兩支麥克風都在中間，CORR 永遠是 +1，看不出問題。", "p"),
    ("MIX 18 PAN 打 LEFT、MIX 19 PAN 打 RIGHT", "n"),
    ("第一支麥克風送 MIX 18、第二支送 MIX 19", "n"),
    ("送出點：要看節目實際混出來的樣子用 POST；推桿還沒推上（彩排、上線前）用 PRE", "li2"),
    ("MONITOR DEFINE 開兩個 BUS：MIX 18＋MIX 19", "n"),
    ("先驗證：只開第一支麥克風，Loudness 畫面應該只有 L 在跳", "warn"),
    ("看完切回：DEFINE 改回 MIX 20，或關掉 DEFINE、改開 STEREO A", "warn"),
]
# 每個畫面前的簡介卡：(畫面編號, 名稱, 一句話, 重點, 縮圖來源 (素材名, 幀))
ABOUT = {
    "about_loudness": ("畫面 1", "LOUDNESS", "現在、這 3 秒、這一小時有多大聲", [
        "平常監看用的主畫面，開機就是這一頁",
        "數字單位都是 LUFS，越接近 0 越大聲",
        "−23 是廣播的目標值（整段節目的平均）"], ("loudness", 20)),
    "about_spectrum": ("畫面 2", "SPECTRUM", "聲音裡哪些頻率比較多", [
        "把聲音拆成 30 個頻段，從 25 Hz 到 20 kHz",
        "找出刺耳、轟轟作響或想消除的頻率",
        "Pi 只負責看，EQ 修正在混音台上做"], ("spectrum", 20)),
    "about_histogram": ("畫面 3", "HISTOGRAM", "過去一段時間的音量，排成一條時間軸", [
        "左舊右新，一眼看出哪一段太大聲、太小聲",
        "整體有沒有貼著目標值",
        "兩種範圍：這一小時／最近 3 分鐘"], ("histogram", 10)),
    "about_phase": ("畫面 4", "PHASE", "兩支麥克風有沒有互相抵消", [
        "兩支靠得近、一起推上去：聲音互相抵消一部分，聽起來悶、空（梳狀濾波）",
        "線材 Pin 2／3 接反：整個抵消（極性接反）",
        "兩頁：GONIO 看極性、BAND 看悶不悶"], ("phase_gonio", 290)),   # 切到正常擺位 3 秒後，MIN 已不是反相留下的 −0.99
}
QR_CARD = ROOT / "qr_card.png"
REMOTE_CARD_QR = [
    ("沒有混音台按鍵時：手機遙控", "h"),
    ("Pi 自己開了 Wi-Fi 熱點，手機連上就能切畫面", "chain"),
    ("手機掃下方 ① 連上 Pi 的 Wi-Fi 熱點", "n"),
    ("掃 ② 打開遙控頁", "n"),
    ("下排四顆直接跳到畫面；上排「子頁」＝ KEY 10／11", "n"),
    ("右上角「？說明」打開各畫面的說明", "n"),
    ("手機每 1.5 秒同步：按了混音台的 KEY 9，手機也跟著變", "li2"),
]
REMOTE_CARD = [
    ("沒有混音台按鍵時：手機遙控", "h"),
    ("Pi 自己開了 Wi-Fi 熱點，手機連上就能切畫面", "chain"),
    ("手機連 Pi 的 Wi-Fi 熱點（掃 Pi 盒子上 QR 卡的 ①）", "n"),
    ("瀏覽器開遙控頁（掃 QR 卡的 ②）", "n"),
    ("下排四顆直接跳到畫面；上排「子頁」＝ KEY 10／11", "n"),
    ("右上角「？說明」打開各畫面的說明", "n"),
    ("手機每 1.5 秒同步：按了混音台的 KEY 9，手機也跟著變", "li2"),
]
CHECK_CARD = ([MK_DEFINE, MK_LEVEL, MK_ON, MK_KEY12], [
    ("讀數出不來？依序檢查", "h"),
    ("平常都已設定好；出不來代表某一環被動到", "chain2"),
    ("MIX 20：ON，輸出電平 0 dB", "n"),
    ("（按 MIX MASTER 後轉 MIX 編碼器，或在畫面上設定）", "li2"),
    ("音軌送 MIX 20：SEND ON（PRE／POST 皆可）", "n"),
    ("要不受推桿影響、單獨看某個音軌 → 用 PRE", "li2"),
    ("MONITOR OUT：ON，畫面上 MONITOR LEVEL 0 dB", "n"),
    ("＋面板 LEVEL 旋鈕在校正位置（兩者一起決定大小）", "li2"),
    ("MONITOR 選 DEFINE，關掉 STEREO A", "n"),
    ("DEFINE 指定 MIX 20（MON/CUE 畫面，或按 KEY 12）", "n"),
])
INTRO = [4.0, 9.0, 9.0, 9.0]     # 片頭四張卡各幾秒（標題、訊號、檢查、手機遙控）
OUTRO = 6.0
OUTRO_LINKS = [("功能詳細介紹與原始碼", "https://github.com/AzureRush/lufsmeterforpi"),
               ("讓 Audio Meter「有意義的」跳動", "https://hackmd.io/@penomadd/audiometer")]
TOTAL = sum(INTRO) + sum(s[1] for s in SEGMENTS) + OUTRO


def font(size, weight=400):
    f = ImageFont.truetype(FONT, size)
    f.set_variation_by_axes([weight])
    return f


F_TITLE, F_SUB, F_TAG, F_NOTE = font(50, 700), font(34), font(28, 700), font(26)
F_KEY = font(28, 500)
F_BIG, F_MID = font(84, 800), font(44, 500)


def load_gif(name):
    if name.startswith("phase_"):   # PHASE：逐幀 PNG，用到才讀（300 張全載入要 800 MB）
        paths = sorted((ROOT / "intro_src" / "phase").glob(name[6:] + "_*.png"))
        if not paths:
            sys.exit("缺 PHASE 畫面：先跑 tools/render_phase_frames.py")
        return Frames(paths)
    return [f.convert("RGB") for f in ImageSequence.Iterator(Image.open(ROOT / "assets" / f"{name}_preview.gif"))]


class Frames:
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    @functools.lru_cache(maxsize=4)
    def __getitem__(self, i):
        return Image.open(self.paths[i]).convert("RGB")


def selected_bar(frame):
    """Spectrum 選定頻段（淺藍色長條）的 x 範圍，從長條底部那一列找。"""
    xs = [x for x in range(258, 1280)
          if (lambda p: p[2] > 190 and p[0] < 120 and 140 < p[1] < 185)(frame.getpixel((x, 710)))]
    return (min(xs) - 4, 50, max(xs) + 4, 720) if xs else None


def to_canvas(r):
    return (GX + r[0] * SC, GY + r[1] * SC, GX + r[2] * SC, GY + r[3] * SC)


def resolve(rects, frame):
    out = [selected_bar(frame) if r == SELECTED else r for r in rects]
    return [to_canvas(r) for r in out if r]


def dim_mask(rects):
    m = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(m)
    d.rectangle((GX, GY, GX + GW - 1, GY + GH - 1), fill=DIM)
    for r in rects:
        d.rectangle(r, fill=0)
    return m


def fit(path, max_w, max_h):
    im = Image.open(path).convert("RGB")
    s = min(max_w / im.width, max_h / im.height)
    return im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS), s


def title_card():
    """片頭第 1 張：標題＋這支影片講什麼＋現場照片。"""
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((W / 2, 70), TITLE, font=F_BIG, fill=(240, 246, 252), anchor="mt")
    d.text((W / 2, 190), "訊號哪來　·　4 個畫面怎麼看　·　怎麼切換", font=F_MID, fill=ACCENT, anchor="mt")
    src = Image.open(PHOTO1).convert("RGB")
    for box in PHOTO1_MOSAIC:   # 馬賽克：縮到 1/40 再放大回去
        region = src.crop(box)
        small = region.resize((max(1, region.width // 40), max(1, region.height // 40)), Image.BILINEAR)
        src.paste(small.resize(region.size, Image.NEAREST), box[:2])
    s = min(1500 / src.width, 780 / src.height)
    photo = src.resize((round(src.width * s), round(src.height * s)), Image.LANCZOS)
    img.paste(photo, ((W - photo.width) // 2, 270))
    return img


def panel_card(spec):
    """片頭第 2、3 張：左邊 PM5D 面板照片（框出相關按鍵）＋右邊文字。"""
    marks, lines = spec
    img = Image.new("RGB", (W, H), BG)
    photo, s = fit(PHOTO2, 760, 1040)
    px, py = 80, (H - photo.height) // 2
    img.paste(photo, (px, py))
    d = ImageDraw.Draw(img)
    for (x0, y0, x1, y1), label in marks:
        r = (px + x0 * s, py + y0 * s, px + x1 * s, py + y1 * s)
        d.rectangle(r, outline=ACCENT, width=5)
        if not label:
            continue
        # 標籤墊深色底，壓在照片上也看得清楚
        lx, ly = r[2] + 12, (r[1] + r[3]) / 2
        bx = d.textbbox((lx, ly), label, font=F_TAG, anchor="lm")
        d.rounded_rectangle((bx[0] - 10, bx[1] - 6, bx[2] + 10, bx[3] + 6), 8, fill=BG, outline=ACCENT, width=2)
        d.text((lx, ly), label, font=F_TAG, fill=(240, 246, 252), anchor="lm")
    draw_lines(d, px + photo.width + 90, 150, lines)
    return img


def about_card(spec, thumb):
    """畫面簡介卡：左邊文字、右邊該畫面縮圖。"""
    num, name, one, points, _ = spec
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    x, tw = 120, 760                      # 文字區寬 760，右邊縮圖從 x=1000 起，中間留 120
    d.text((x, 230), num, font=F_TAG, fill=ACCENT)
    d.text((x, 270), name, font=F_BIG, fill=(240, 246, 252))
    y = _wrap(d, x, 400, one, F_MID, tw, ACCENT, 60) + 50
    for p in points:
        d.text((x, y), "•", font=F_SUB, fill=ACCENT)
        y = _wrap(d, x + 46, y, p, F_SUB, tw - 46, (201, 209, 217)) + 22
    t = thumb.resize((800, 450), Image.LANCZOS)
    tx, ty = W - 120 - t.width, (H - t.height) // 2
    img.paste(t, (tx, ty))
    d.rectangle((tx - 2, ty - 2, tx + t.width + 1, ty + t.height + 1), outline=(48, 54, 61), width=2)
    return img


def _wrap(d, x, y, text, f, max_w, fill, lh=52):
    """中文逐字折行，回傳下一行的 y。"""
    line = ""
    for ch in text:
        if d.textlength(line + ch, font=f) > max_w and ch not in "，。、；：）」":   # 標點不放行首
            d.text((x, y), line, font=f, fill=fill)
            y, line = y + lh, ""
        line += ch
    d.text((x, y), line, font=f, fill=fill)
    return y + lh


def remote_card():
    """片頭第 4 張：手機遙控頁截圖＋步驟；有 qr_card.png 就在文字下方放整張 QR 卡（含熱點密碼，見檔頭）。"""
    img = Image.new("RGB", (W, H), BG)
    phone, _ = fit(ROOT / "intro_src" / "remote_phone.png", 480, 960)
    px, py = 160, (H - phone.height) // 2
    img.paste(phone, (px, py))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((px - 14, py - 14, px + phone.width + 14, py + phone.height + 14), 28, outline=(72, 79, 88), width=6)
    tx = px + phone.width + 120
    if not QR_CARD.exists():
        draw_lines(d, tx, 170, REMOTE_CARD)
        return img
    draw_lines(d, tx, 90, REMOTE_CARD_QR)
    qr, _ = fit(QR_CARD, 760, 420)   # 945×591 → 約 0.71 倍，QR 每格約 7 px，螢幕上掃得到
    img.paste(qr, (tx, 1080 - 40 - qr.height))
    return img


def text_card(lines):
    """沒照片的字卡（PHASE 前提）：文字置中偏左排。"""
    img = Image.new("RGB", (W, H), BG)
    draw_lines(ImageDraw.Draw(img), 300, 150, lines)
    return img


def draw_lines(d, x, y, lines):
    style = {"h": (F_BIG, (240, 246, 252), 0, 130), "chain": (F_MID, ACCENT, 0, 110),
             "chain2": (F_SUB, ACCENT, 0, 80),
             "p": (F_SUB, (201, 209, 217), 0, 56), "li": (F_SUB, (240, 246, 252), 0, 56),
             "n": (F_SUB, (240, 246, 252), 0, 56), "li2": (F_SUB, (201, 209, 217), 46, 56),
             "warn": (F_SUB, (220, 200, 50), 0, 56)}
    prev, num = None, 0
    for text, kind in lines:
        f, color, indent, step = style[kind]
        if kind in ("li", "warn", "n") and prev == "p" or kind == "warn" and prev != "warn":
            y += 36           # 段落之間空一點
        if kind == "li":
            d.text((x, y), "•", font=f, fill=ACCENT)
            indent = 46
        elif kind == "n":
            num += 1
            d.text((x, y), f"{num}", font=F_TAG, fill=ACCENT)
            indent = 46
        elif kind == "warn":
            d.text((x, y), "⚠", font=f, fill=color)
            indent = 46
        d.text((x + indent, y), text, font=f, fill=color)
        y += step
        prev = kind


def segment_frame(gif, seg, t):
    _, dur, (a, b), tag, keys, steps = seg
    frame = gif[min(b - 1, a + int((b - a) * t / dur))]
    img = Image.new("RGB", (W, H), BG)
    img.paste(frame.resize((GW, GH), Image.LANCZOS), (GX, GY))

    i = max(n for n, s in enumerate(steps) if s[0] <= t)
    start, rects, title, sub = steps[i]
    cur = resolve(rects, frame)
    # 步驟之間交叉淡化兩組框選區（框外一直保持調暗，不會整片先亮再暗）；段落第一步直接到位
    prev = resolve(steps[i - 1][1], frame) if i else []
    k = min(1.0, (t - start) / FADE) if i and prev != cur else 1.0
    mask = dim_mask(cur) if k >= 1 else Image.blend(dim_mask(prev), dim_mask(cur), k)
    img.paste((0, 0, 0), (0, 0, W, H), mask)

    lines = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ld = ImageDraw.Draw(lines)
    for rs, alpha in ((prev, 1 - k), (cur, k)):
        for r in rs:
            ld.rectangle(r, outline=ACCENT + (int(255 * alpha),), width=5)
    img = Image.alpha_composite(img.convert("RGBA"), lines).convert("RGB")

    d = ImageDraw.Draw(img)
    cy = GY + GH + 14
    d.text((GX, cy), tag, font=F_TAG, fill=ACCENT)
    d.text((GX + GW, cy), keys, font=F_KEY, fill=(139, 148, 158), anchor="ra")
    d.text((GX, cy + 38), title, font=F_TITLE, fill=(240, 246, 252))
    if sub == "legend":
        x = GX
        for color, label in COLOR_LEGEND:
            d.rectangle((x, cy + 124, x + 34, cy + 154), fill=color)
            d.text((x + 46, cy + 139), label, font=F_SUB, fill=(201, 209, 217), anchor="lm")
            x += 46 + d.textlength(label, font=F_SUB) + 56
        # 附註放在上半部被調暗的區域，不擋到框選內容
        tw = d.textlength(COLOR_NOTE, font=F_NOTE)
        x, y = GX + 40, GY + 200
        d.rounded_rectangle((x - 20, y - 16, x + tw + 20, y + 46), 10, fill=(22, 27, 34), outline=(48, 54, 61))
        d.text((x, y), COLOR_NOTE, font=F_NOTE, fill=(230, 237, 243))
    else:
        d.text((GX, cy + 116), sub, font=F_SUB, fill=(139, 148, 158))
    return img


def build():
    gifs = {s[0]: load_gif(s[0]) for s in SEGMENTS if s[0] != "card"}
    seg_cards = {"phase_card": text_card(PHASE_CARD)}
    for k, spec in ABOUT.items():
        src, i = spec[4]
        seg_cards[k] = about_card(spec, gifs[src][i])
    cards = [title_card(), panel_card(SIGNAL_CARD), panel_card(CHECK_CARD), remote_card()]

    def frame_at(t):
        for card, dur in zip(cards, INTRO):
            if t < dur:
                return card
            t -= dur
        for seg in SEGMENTS:
            if t < seg[1]:
                return seg_cards[seg[2]] if seg[0] == "card" else segment_frame(gifs[seg[0]], seg, t)
            t -= seg[1]
        return outro

    outro = outro_card()
    return frame_at


def outro_card():
    """片尾：GitHub repo 與 HackMD 介紹文章的 QR code。"""
    import segno   # pip install segno（純 Python、無相依）
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((W / 2, 110), TITLE, font=F_BIG, fill=(240, 246, 252), anchor="mt")
    size, gap = 520, 260
    x0 = (W - size * 2 - gap) // 2
    for i, (label, url) in enumerate(OUTRO_LINKS):
        buf = __import__("io").BytesIO()
        segno.make(url, error="m").save(buf, kind="png", scale=20, border=2, dark="#000000", light="#ffffff")
        qr = Image.open(buf).convert("RGB").resize((size, size), Image.NEAREST)
        x = x0 + i * (size + gap)
        img.paste(qr, (x, 300))
        d.text((x + size / 2, 300 + size + 40), label, font=F_TITLE, fill=ACCENT, anchor="mt")
        d.text((x + size / 2, 300 + size + 110), url.split("://")[1], font=F_NOTE, fill=(139, 148, 158), anchor="mt")
    return img


def main():
    frame_at = build()
    if "--still" in sys.argv:
        for s in sys.argv[sys.argv.index("--still") + 1:]:
            p = ROOT / f"intro_still_{s}.png"
            frame_at(float(s)).save(p)
            print(p)
        return
    out = ROOT / "intro.mp4"
    ff = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "20",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    for i in range(int(TOTAL * FPS)):
        ff.stdin.write(frame_at(i / FPS).tobytes())
    ff.stdin.close()
    if ff.wait():
        sys.exit("ffmpeg failed")
    print(out, f"{TOTAL:.0f}s")


if __name__ == "__main__":
    main()
