"""音訊輸入裝置選擇的自我檢查（不需要硬體）：指定名稱／Scarlett／沒 Scarlett 且無鍵盤／完全沒有輸入裝置。
重點是「開機自動啟動時沒有鍵盤」那條路：以前會卡在 input() 的無限迴圈。"""
import os
import sys
from unittest import mock

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "lufs_core", "target", "release"))   # 筆電上用 cargo build 出來的 .pyd
for name in ("sounddevice", "pygame", "gpiozero"):
    sys.modules[name] = mock.MagicMock()
try:
    import scipy.signal  # noqa: F401
except ImportError:
    sys.modules["scipy"] = mock.MagicMock()
    sys.modules["scipy.signal"] = mock.MagicMock()

import loudness_meter as lm  # noqa: E402


def dev(name, ins):
    return {"name": name, "max_input_channels": ins}


HDMI = dev("vc4-hdmi-0: MAI PCM i2s-hifi-0 (hw:0,0)", 0)          # 只有輸出
MIC = dev("USB PnP Sound Device: Audio (hw:2,0)", 1)                # 單聲道
UMC = dev("UMC204HD 192k: USB Audio (hw:3,0)", 2)
SCAR = dev("Scarlett 2i2 USB: Audio (hw:4,0)", 2)
no_input = mock.patch("builtins.input", side_effect=AssertionError("不該問使用者"))

with no_input:
    assert lm.select_input_device([HDMI, UMC, SCAR]) == 2, "有 Scarlett 就選它"
    assert lm.select_input_device([HDMI, UMC, SCAR], want="umc") == 1, "LM_AUDIO_DEVICE 優先於 Scarlett"
    assert lm.select_input_device([HDMI, UMC, SCAR], want="behringer") == 2, "指定的找不到 → 退回 Scarlett"
    assert lm.select_input_device([HDMI, MIC, UMC]) == 2, "沒 Scarlett、無鍵盤 → 自動選第一個 2 聲道（跳過單聲道麥克風）"
    assert lm.select_input_device([HDMI, MIC]) == 1, "沒有 2 聲道的 → 退而選第一個輸入裝置"
print("[OK] 指定名稱／Scarlett／無鍵盤自動選，都不會停下來問")

try:
    lm.select_input_device([HDMI])
    raise AssertionError("沒有輸入裝置應該結束程式")
except SystemExit:
    print("[OK] 沒有任何輸入裝置 → 結束程式（不卡住）")

with mock.patch("builtins.input", side_effect=["x", "9", "2"]):
    assert lm.select_input_device([HDMI, MIC, UMC], interactive=True) == 2, "有鍵盤 → 照使用者選的第 2 個"
with mock.patch("builtins.input", side_effect=EOFError):
    try:
        lm.select_input_device([HDMI, MIC, UMC], interactive=True)
        raise AssertionError("Ctrl-D 應該結束程式")
    except SystemExit:
        pass
print("[OK] 有鍵盤時照舊詢問；輸入錯誤會重問，Ctrl-D 結束（不再無限迴圈）")
print("\n全部通過")
