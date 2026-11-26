"""手機網頁自我檢查：起 server 在隨機 port，用 urllib 打各端點，assert 狀態改變正確。

    python tools/check_web.py      # 需 numpy、pygame；筆電沒有 sounddevice 等模組時自動塞空殼
"""
import os, sys, types, json, urllib.request
os.environ["SDL_VIDEODRIVER"] = "dummy"; os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["LM_WEB_PORT"] = "0"   # 先確認 0 = 停用
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
try:
    import sounddevice, gpiozero, lufs_core, scipy.signal  # noqa: F401  Pi 上都有
except ImportError:   # ponytail: 筆電沒有這些模組，塞空殼只為了能 import
    for name in ("sounddevice", "gpiozero", "lufs_core", "scipy", "scipy.signal"):
        sys.modules[name] = types.ModuleType(name)
    sys.modules["gpiozero"].Button = object
    sys.modules["lufs_core"].EnergyStore = lambda n: None
    sys.modules["scipy.signal"].lfilter = None
import loudness_meter as lm

assert lm.start_web_server() is None, "LM_WEB_PORT=0 應停用"
os.environ["LM_WEB_PORT"] = "abc"
assert lm.start_web_server() is None, "port 格式錯應停用、不崩潰"
os.environ["LM_WEB_PORT"] = "18080"   # 0 在這支程式代表停用，所以測試用固定的高 port
srv = lm.start_web_server()
base = f"http://127.0.0.1:{srv.server_address[1]}"


def req(path, post=False):
    with urllib.request.urlopen(urllib.request.Request(base + path, data=b"" if post else None, method="POST" if post else "GET"), timeout=3) as r:
        return r.read().decode()


html = req("/")
assert "LOUD" in html and "/state" in html and 'href="/help"' in html

h = req("/help")                                                                   # README → 說明頁
assert "畫面四" in h and "<table>" in h and 'id="fn-lufs"' in h, h[:300]
assert "GPI 訊號對應" not in h and "技術架構" not in h and "更新紀錄" not in h       # 只放「怎麼看」
assert "QR 卡" not in h and "LM_WEB_PORT" not in h and "LM_PHASE_LOG" not in h and "setup_hotspot" not in h   # help:skip 夾起來的
assert "MIX 18" in h and "help:skip" not in h and 'src="/assets/phase_band.png"' in h
assert "phase_gonio.png" in html                                                    # 遙控頁範例圖
assert 'src="/assets/loudness_preview.gif"' in h and "githubusercontent" not in h
assert "](" not in h and "**" not in h                                             # 沒有漏轉的 Markdown
with urllib.request.urlopen(base + "/assets/loudness_preview.gif", timeout=3) as r:
    assert r.headers["Content-Type"] == "image/gif" and r.read(6) == b"GIF89a"
for bad in ("/assets/../loudness_meter.py", "/assets/..", "/assets/", "/assets/nope.gif"):
    try:
        req(bad); raise AssertionError(bad + " 應回 404")
    except urllib.error.HTTPError as e:
        assert e.code == 404
s = json.loads(req("/state"))
assert s["mode"] == "loudness", s

for m in ("phase", "histogram", "spectrum", "loudness"):
    assert json.loads(req("/mode/" + m, True))["mode"] == m == lm.display_mode

assert json.loads(req("/next", True))["mode"] == "loudness" and lm.freq_index == 0   # Loudness：KEY 10/11 不作用
req("/mode/spectrum", True)
assert json.loads(req("/next", True))["freq"] == "31.5 Hz"                         # 往高頻一格
assert json.loads(req("/prev", True))["freq"] == "25 Hz"
assert json.loads(req("/prev", True))["freq"] == "ALL"                             # 繞到 BYPASS
req("/mode/histogram", True)
assert json.loads(req("/next", True))["hist_mode"] == "slide"
req("/mode/phase", True)
assert json.loads(req("/prev", True))["phase_view"] == "band"
assert json.loads(req("/next", True))["phase_view"] == "gonio"

lm._toggle_display_mode()                                                          # 實體 KEY 9：網頁輪詢要看得到
assert json.loads(req("/state"))["mode"] == "loudness"

for bad in ("/mode/xyz", "/nope"):
    try:
        req(bad, True); raise AssertionError(bad + " 應回 404")
    except urllib.error.HTTPError as e:
        assert e.code == 404
assert lm.display_mode == "loudness"
srv.shutdown()
print("網頁檢查全部通過")
