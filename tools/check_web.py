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
assert "Loudness" in html and "/state" in html
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
