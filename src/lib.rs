#![warn(clippy::all)]
#![allow(clippy::borrow_deref_ref)]
use feos::pcsaft::{PcSaft, PcSaftParameters, PcSaftRecord};
use feos_core::parameter::{Parameter, PureRecord};
use feos_core::{DensityInitialization, PhaseEquilibrium, State};
use ndarray::{arr1, Array1, Array2, ArrayView1, Zip};
use numpy::{PyArray1, PyArray2, ToPyArray};
use pyo3::prelude::*;
use quantity::si::{ANGSTROM, KELVIN, MOL, NAV, PASCAL};
use std::rc::Rc;

#[pyclass]
struct PcSaftParallel {
    parameter: Array2<f64>,
}

#[pymethods]
impl PcSaftParallel {
    #[new]
    pub fn new(parameter: &PyArray2<f64>) -> Self {
        Self {
            parameter: parameter.to_owned_array(),
        }
    }

    fn vapor_pressure<'py>(
        &self,
        temperature: &PyArray1<f64>,
        py: Python<'py>,
    ) -> &'py PyArray2<f64> {
        self.vapor_pressure_(temperature.readonly().as_array())
            .view()
            .to_pyarray(py)
    }

    fn liquid_density<'py>(
        &self,
        temperature: &PyArray1<f64>,
        pressure: &PyArray1<f64>,
        py: Python<'py>,
    ) -> &'py PyArray1<f64> {
        self.liquid_density_(
            temperature.readonly().as_array(),
            pressure.readonly().as_array(),
        )
        .view()
        .to_pyarray(py)
    }
}

impl PcSaftParallel {
    fn vapor_pressure_(&self, temperature: ArrayView1<f64>) -> Array2<f64> {
        let mut rho = Array2::zeros([temperature.len(), 2]);
        Zip::from(rho.rows_mut())
            .and(self.parameter.rows())
            .and(&temperature)
            .par_for_each(|mut rho, par, &t| {
                let eos = build_eos(par);
                let vle = PhaseEquilibrium::pure(&eos, t * KELVIN, None, Default::default());
                rho[0] = match &vle {
                    Err(_) => f64::NAN,
                    Ok(vle) => vle
                        .vapor()
                        .density
                        .to_reduced(ANGSTROM.powi(-3) / NAV)
                        .unwrap(),
                };
                rho[1] = match &vle {
                    Err(_) => f64::NAN,
                    Ok(vle) => vle
                        .liquid()
                        .density
                        .to_reduced(ANGSTROM.powi(-3) / NAV)
                        .unwrap(),
                };
            });
        rho
    }

    fn liquid_density_(
        &self,
        temperature: ArrayView1<f64>,
        pressure: ArrayView1<f64>,
    ) -> Array1<f64> {
        Zip::from(self.parameter.rows())
            .and(&temperature)
            .and(&pressure)
            .par_map_collect(|par, &t, &p| {
                let eos = build_eos(par);
                let state = State::new_npt(
                    &eos,
                    t * KELVIN,
                    p * PASCAL,
                    &(arr1(&[1.0]) * MOL),
                    DensityInitialization::Liquid,
                );
                match &state {
                    Err(_) => f64::NAN,
                    Ok(state) => state.density.to_reduced(ANGSTROM.powi(-3) / NAV).unwrap(),
                }
            })
    }
}

fn build_eos(parameter: ArrayView1<f64>) -> Rc<PcSaft> {
    let record = PcSaftRecord::new(
        parameter[0],
        parameter[1],
        parameter[2],
        Some(parameter[3]),
        None,
        Some(parameter[4]),
        Some(parameter[5]),
        None,
        None,
        None,
        None,
        None,
    );
    let record = PureRecord::new(Default::default(), 0.0, record, None);
    let params = Rc::new(PcSaftParameters::new_pure(record));
    Rc::new(PcSaft::new(params))
}

#[pymodule]
pub fn feos_torch(_: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    m.add_class::<PcSaftParallel>()
}
