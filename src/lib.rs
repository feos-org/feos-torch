#![warn(clippy::all)]
#![allow(clippy::borrow_deref_ref)]
use pyo3::prelude::*;

mod gc_pcsaft;
mod pcsaft;
use gc_pcsaft::PyGcPcSaft;
use pcsaft::PyPcSaft;

#[pymodule]
pub fn feos_torch(_: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    m.add_class::<PyPcSaft>()?;
    m.add_class::<PyGcPcSaft>()
}
