use pyo3::prelude::*;
use std::collections::VecDeque;
use std::sync::{Mutex, MutexGuard};

/// 對應 Python 版 _energy_to_lufs()：-0.691 + 10*log10(e)
fn energy_to_lufs(e: f64) -> f64 {
    -0.691 + 10.0 * e.log10()
}

/// EBU R128 two-stage gated integrated loudness，逐 block 能量算出 THIS HOUR / SEGMENT。
///
/// `relative_gate_lu`：相對閾值的 LU 數。ITU-R BS.1770-5 Annex 1 官方值是 10.0——
/// 「the second at −10 dB relative to the level measured after application of
/// the first threshold」。舊 Python 版 compute_h() 誤植成 8.0（2026-09-19 已在
/// loudness_meter.py 改為 10.0，見 implementation-notes.md），這裡保留成參數而不寫死，
/// 讓對照腳本能同時驗證兩個值。
fn gated_lufs(block_energies: &[[f64; 2]], relative_gate_lu: f64) -> f64 {
    if block_energies.len() < 2 {
        return -70.0;
    }

    let power: Vec<f64> = block_energies
        .iter()
        .map(|e| (e[0] + e[1]).max(1e-12))
        .collect();
    let lufs: Vec<f64> = power.iter().map(|&p| energy_to_lufs(p)).collect();

    // 第一階段：絕對閾值 -70 LUFS
    let gate1: Vec<f64> = power
        .iter()
        .zip(&lufs)
        .filter(|&(_, &l)| l > -70.0)
        .map(|(&p, _)| p)
        .collect();
    if gate1.is_empty() {
        return -70.0;
    }
    let mean1 = gate1.iter().sum::<f64>() / gate1.len() as f64;
    let rel = energy_to_lufs(mean1) - relative_gate_lu;

    // 第二階段：相對閾值
    let gate2: Vec<f64> = power
        .iter()
        .zip(&lufs)
        .filter(|&(_, &l)| l > rel)
        .map(|(&p, _)| p)
        .collect();
    if gate2.is_empty() {
        return -70.0;
    }
    let mean2 = gate2.iter().sum::<f64>() / gate2.len() as f64;
    energy_to_lufs(mean2)
}

/// 純函式版：吃一整份 Python list，供對照腳本使用。
/// 正式程式不走這條——每次把 72000 筆 list-of-list 轉成 Rust 資料就要卡住 Python
/// 約 20ms（Pi 4B 實測），跟它省下的計算時間差不多，見 EnergyStore。
#[pyfunction]
#[pyo3(signature = (block_energies, relative_gate_lu=10.0))]
fn compute_h(block_energies: Vec<[f64; 2]>, relative_gate_lu: f64) -> f64 {
    gated_lufs(&block_energies, relative_gate_lu)
}

/// 每個 block 的 [E_L, E_R] 能量，由 Rust 保管——語意等同 Python 的
/// `deque(maxlen=cap)`（滿了就擠掉最舊的）。
///
/// 為什麼不繼續用 Python deque：compute_h 要算的資料在 Python 那邊，每次計算都得先
/// 把它整份轉成 Rust／numpy 資料，這步一定持有 GIL、成本隨累積量線性成長（72000 筆
/// ≈ 50ms），期間音訊 callback 拿不到 GIL 就 input overflow。資料放在 Rust 這邊，
/// callback 每個 block 只 push 兩個浮點數（微秒級），compute_h 只在持鎖複製一份快照
/// 時佔著 GIL，真正的運算在釋放 GIL 之後進行。
#[pyclass(frozen)]
struct EnergyStore {
    cap: usize,
    buf: Mutex<VecDeque<[f64; 2]>>,
}

impl EnergyStore {
    /// 不讓 Mutex poison 變成 audio callback 裡的例外——callback 一丟例外，stream 就停了。
    fn lock(&self) -> MutexGuard<'_, VecDeque<[f64; 2]>> {
        self.buf.lock().unwrap_or_else(|e| e.into_inner())
    }
}

#[pymethods]
impl EnergyStore {
    #[new]
    fn new(cap: usize) -> Self {
        Self {
            cap,
            buf: Mutex::new(VecDeque::with_capacity(cap)),
        }
    }

    fn push(&self, l: f64, r: f64) {
        let mut b = self.lock();
        if b.len() == self.cap {
            b.pop_front();
        }
        b.push_back([l, r]);
    }

    fn clear(&self) {
        self.lock().clear();
    }

    fn __len__(&self) -> usize {
        self.lock().len()
    }

    #[pyo3(signature = (relative_gate_lu=10.0))]
    fn compute_h(&self, py: Python<'_>, relative_gate_lu: f64) -> f64 {
        // 持鎖只做一次複製（72000 筆約 1MB）；此時 GIL 也在呼叫端手上，
        // audio callback 不可能同時 push，不會互相卡。
        let snapshot: Vec<[f64; 2]> = self.lock().iter().copied().collect();
        py.detach(|| gated_lufs(&snapshot, relative_gate_lu))
    }
}

#[pymodule]
fn lufs_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_h, m)?)?;
    m.add_class::<EnergyStore>()?;
    Ok(())
}
