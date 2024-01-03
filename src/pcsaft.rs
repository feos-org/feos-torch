#![warn(clippy::all)]
#![allow(clippy::borrow_deref_ref)]
use feos::core::parameter::{Parameter, PureRecord};
use feos::core::si::{KELVIN, MOL, PASCAL};
use feos::core::{DensityInitialization, PhaseEquilibrium, State};
use feos::pcsaft::{PcSaft, PcSaftBinaryRecord, PcSaftParameters, PcSaftRecord};
use ndarray::{arr1, s, Array1, Array2, ArrayView1, ArrayView2, ArrayView3, Zip};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, ToPyArray};
use pyo3::prelude::*;
use std::sync::Arc;

#[pyclass]
pub struct PcSaftParallel;

#[pymethods]
impl PcSaftParallel {
    #[staticmethod]
    fn vapor_pressure<'py>(
        parameters: PyReadonlyArray2<f64>,
        temperature: PyReadonlyArray1<f64>,
        py: Python<'py>,
    ) -> &'py PyArray2<f64> {
        vapor_pressure_(parameters.as_array(), temperature.as_array())
            .view()
            .to_pyarray(py)
    }

    #[staticmethod]
    fn liquid_density<'py>(
        parameters: PyReadonlyArray2<f64>,
        temperature: PyReadonlyArray1<f64>,
        pressure: PyReadonlyArray1<f64>,
        py: Python<'py>,
    ) -> &'py PyArray1<f64> {
        liquid_density_(
            parameters.as_array(),
            temperature.as_array(),
            pressure.as_array(),
        )
        .view()
        .to_pyarray(py)
    }

    #[staticmethod]
    fn bubble_point<'py>(
        parameters: PyReadonlyArray3<f64>,
        kij: PyReadonlyArray2<f64>,
        temperature: PyReadonlyArray1<f64>,
        liquid_molefracs: PyReadonlyArray1<f64>,
        pressure: PyReadonlyArray1<f64>,
        py: Python<'py>,
    ) -> &'py PyArray2<f64> {
        bubble_point_(
            parameters.as_array(),
            kij.as_array(),
            temperature.as_array(),
            liquid_molefracs.as_array(),
            pressure.as_array(),
        )
        .view()
        .to_pyarray(py)
    }

    #[staticmethod]
    fn dew_point<'py>(
        parameters: PyReadonlyArray3<f64>,
        kij: PyReadonlyArray2<f64>,
        temperature: PyReadonlyArray1<f64>,
        vapor_molefracs: PyReadonlyArray1<f64>,
        pressure: PyReadonlyArray1<f64>,
        py: Python<'py>,
    ) -> &'py PyArray2<f64> {
        dew_point_(
            parameters.as_array(),
            kij.as_array(),
            temperature.as_array(),
            vapor_molefracs.as_array(),
            pressure.as_array(),
        )
        .view()
        .to_pyarray(py)
    }
}

fn vapor_pressure_(parameters: ArrayView2<f64>, temperature: ArrayView1<f64>) -> Array2<f64> {
    let mut rho = Array2::zeros([temperature.len(), 2]);
    Zip::from(rho.rows_mut())
        .and(parameters.rows())
        .and(&temperature)
        .par_for_each(|mut rho, par, &t| {
            let params = PcSaftParameters::new_pure(build_record(par)).unwrap();
            let eos = Arc::new(PcSaft::new(Arc::new(params)));
            let vle = PhaseEquilibrium::pure(&eos, t * KELVIN, None, Default::default());
            match vle {
                Err(_) => rho.fill(f64::NAN),
                Ok(vle) => {
                    rho[0] = vle.vapor().density.to_reduced();
                    rho[1] = vle.liquid().density.to_reduced();
                }
            };
        });
    rho
}

fn liquid_density_(
    parameters: ArrayView2<f64>,
    temperature: ArrayView1<f64>,
    pressure: ArrayView1<f64>,
) -> Array1<f64> {
    Zip::from(parameters.rows())
        .and(&temperature)
        .and(&pressure)
        .par_map_collect(|par, &t, &p| {
            let params = PcSaftParameters::new_pure(build_record(par)).unwrap();
            let eos = Arc::new(PcSaft::new(Arc::new(params)));
            let state = State::new_npt(
                &eos,
                t * KELVIN,
                p * PASCAL,
                &(arr1(&[1.0]) * MOL),
                DensityInitialization::Liquid,
            );
            match state {
                Err(_) => f64::NAN,
                Ok(state) => state.density.to_reduced(),
            }
        })
}

fn build_record(parameter: ArrayView1<f64>) -> PureRecord<PcSaftRecord> {
    let record = PcSaftRecord::new(
        parameter[0],
        parameter[1],
        parameter[2],
        Some(parameter[3]),
        None,
        Some(parameter[4]),
        Some(parameter[5]),
        Some(parameter[6]),
        Some(parameter[7]),
        None,
        None,
        None,
        None,
    );
    PureRecord::new(Default::default(), 0.0, record)
}

fn bubble_point_(
    parameters: ArrayView3<f64>,
    kij: ArrayView2<f64>,
    temperature: ArrayView1<f64>,
    liquid_molefracs: ArrayView1<f64>,
    pressure: ArrayView1<f64>,
) -> Array2<f64> {
    let mut rho = Array2::zeros([temperature.len(), 4]);
    Zip::from(rho.rows_mut())
        .and(parameters.outer_iter())
        .and(kij.outer_iter())
        .and(temperature)
        .and(liquid_molefracs)
        .and(pressure)
        .par_for_each(|mut rho, par, kij, &t, &x, &p| {
            let epsilon_k_ab = if kij[1] == 0.0 { None } else { Some(kij[1]) };
            let params = PcSaftParameters::new_binary(
                par.outer_iter().map(build_record).collect(),
                Some(PcSaftBinaryRecord::new(Some(kij[0]), None, epsilon_k_ab)),
            )
            .unwrap();
            let eos = Arc::new(PcSaft::new(Arc::new(params)));
            let vle = PhaseEquilibrium::bubble_point(
                &eos,
                t * KELVIN,
                &arr1(&[x, 1.0 - x]),
                Some(p * PASCAL),
                None,
                Default::default(),
            );
            match vle {
                Err(_) => rho.fill(f64::NAN),
                Ok(vle) => {
                    let rho_v = vle.vapor().partial_density.to_reduced();
                    let rho_l = vle.liquid().partial_density.to_reduced();
                    rho.slice_mut(s![0..2usize]).assign(&rho_v);
                    rho.slice_mut(s![2..4usize]).assign(&rho_l);
                }
            }
        });
    rho
}

fn dew_point_(
    parameters: ArrayView3<f64>,
    kij: ArrayView2<f64>,
    temperature: ArrayView1<f64>,
    vapor_molefracs: ArrayView1<f64>,
    pressure: ArrayView1<f64>,
) -> Array2<f64> {
    let mut rho = Array2::zeros([temperature.len(), 4]);
    Zip::from(rho.rows_mut())
        .and(parameters.outer_iter())
        .and(kij.outer_iter())
        .and(temperature)
        .and(vapor_molefracs)
        .and(pressure)
        .par_for_each(|mut rho, par, kij, &t, &y, &p| {
            let epsilon_k_ab = if kij[1] == 0.0 { None } else { Some(kij[1]) };
            let params = PcSaftParameters::new_binary(
                par.outer_iter().map(build_record).collect(),
                Some(PcSaftBinaryRecord::new(Some(kij[0]), None, epsilon_k_ab)),
            )
            .unwrap();
            let eos = Arc::new(PcSaft::new(Arc::new(params)));
            let vle = PhaseEquilibrium::dew_point(
                &eos,
                t * KELVIN,
                &arr1(&[y, 1.0 - y]),
                Some(p * PASCAL),
                None,
                Default::default(),
            );
            match vle {
                Err(_) => rho.fill(f64::NAN),
                Ok(vle) => {
                    let rho_v = vle.vapor().partial_density.to_reduced();
                    let rho_l = vle.liquid().partial_density.to_reduced();
                    rho.slice_mut(s![0..2usize]).assign(&rho_v);
                    rho.slice_mut(s![2..4usize]).assign(&rho_l);
                }
            }
        });
    rho
}
