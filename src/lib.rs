#![warn(clippy::all)]
#![allow(clippy::borrow_deref_ref)]
use pyo3::prelude::*;

mod gc_pcsaft;
mod pcsaft;
use gc_pcsaft::GcPcSaftParallel;
use pcsaft::PcSaftParallel;

#[pymodule]
pub fn feos_torch(_: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    m.add_class::<PcSaftParallel>()?;
    m.add_class::<GcPcSaftParallel>()
}
