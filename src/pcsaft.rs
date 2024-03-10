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
    ) -> (&'py PyArray2<f64>, &'py PyArray1<bool>) {
        let (rho, status) = vapor_pressure_(parameters.as_array(), temperature.as_array());
        (rho.view().to_pyarray(py), status.view().to_pyarray(py))
    }

    #[staticmethod]
    fn liquid_density<'py>(
        parameters: PyReadonlyArray2<f64>,
        temperature: PyReadonlyArray1<f64>,
        pressure: PyReadonlyArray1<f64>,
        py: Python<'py>,
    ) -> (&'py PyArray1<f64>, &'py PyArray1<bool>) {
        let (rho, status) = liquid_density_(
            parameters.as_array(),
            temperature.as_array(),
            pressure.as_array(),
        );
        (rho.view().to_pyarray(py), status.view().to_pyarray(py))
    }

    #[staticmethod]
    fn bubble_point<'py>(
        parameters: PyReadonlyArray3<f64>,
        kij: PyReadonlyArray2<f64>,
        temperature: PyReadonlyArray1<f64>,
        liquid_molefracs: PyReadonlyArray1<f64>,
        pressure: PyReadonlyArray1<f64>,
        py: Python<'py>,
    ) -> (&'py PyArray2<f64>, &'py PyArray1<bool>) {
        let (rho, status) = bubble_point_(
            parameters.as_array(),
            kij.as_array(),
            temperature.as_array(),
            liquid_molefracs.as_array(),
            pressure.as_array(),
        );
        (rho.view().to_pyarray(py), status.view().to_pyarray(py))
    }

    #[staticmethod]
    fn dew_point<'py>(
        parameters: PyReadonlyArray3<f64>,
        kij: PyReadonlyArray2<f64>,
        temperature: PyReadonlyArray1<f64>,
        vapor_molefracs: PyReadonlyArray1<f64>,
        pressure: PyReadonlyArray1<f64>,
        py: Python<'py>,
    ) -> (&'py PyArray2<f64>, &'py PyArray1<bool>) {
        let (rho, status) = dew_point_(
            parameters.as_array(),
            kij.as_array(),
            temperature.as_array(),
            vapor_molefracs.as_array(),
            pressure.as_array(),
        );
        (rho.view().to_pyarray(py), status.view().to_pyarray(py))
    }
}

fn vapor_pressure_(
    parameters: ArrayView2<f64>,
    temperature: ArrayView1<f64>,
) -> (Array2<f64>, Array1<bool>) {
    let vles = Zip::from(parameters.rows())
        .and(&temperature)
        .par_map_collect(|par, &t| {
            let params = PcSaftParameters::new_pure(build_record(par)).unwrap();
            let eos = Arc::new(PcSaft::new(Arc::new(params)));
            PhaseEquilibrium::pure(&eos, t * KELVIN, None, Default::default()).ok()
        });
    let status = vles.iter().map(|v| v.is_none()).collect();
    let vles: Array1<_> = vles.into_iter().flatten().collect();
    let mut rho = Array2::zeros([vles.len(), 4]);
    Zip::from(rho.rows_mut())
        .and(&vles)
        .for_each(|mut rho, vle| {
            rho[0] = vle.vapor().density.to_reduced();
            rho[1] = vle.liquid().density.to_reduced();
        });
    (rho, status)
}

fn liquid_density_(
    parameters: ArrayView2<f64>,
    temperature: ArrayView1<f64>,
    pressure: ArrayView1<f64>,
) -> (Array1<f64>, Array1<bool>) {
    let states = Zip::from(parameters.rows())
        .and(&temperature)
        .and(&pressure)
        .par_map_collect(|par, &t, &p| {
            let params = PcSaftParameters::new_pure(build_record(par)).unwrap();
            let eos = Arc::new(PcSaft::new(Arc::new(params)));
            State::new_npt(
                &eos,
                t * KELVIN,
                p * PASCAL,
                &(arr1(&[1.0]) * MOL),
                DensityInitialization::Liquid,
            )
            .ok()
        });
    let status = states.iter().map(|v| v.is_none()).collect();
    let states: Array1<_> = states.into_iter().flatten().collect();
    let rho = Zip::from(&states).map_collect(|s| s.density.to_reduced());
    (rho, status)
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
) -> (Array2<f64>, Array1<bool>) {
    let vles = Zip::from(parameters.outer_iter())
        .and(kij.outer_iter())
        .and(temperature)
        .and(liquid_molefracs)
        .and(pressure)
        .par_map_collect(|par, kij, &t, &x, &p| {
            let epsilon_k_ab = if kij[1] == 0.0 { None } else { Some(kij[1]) };
            let params = PcSaftParameters::new_binary(
                par.outer_iter().map(build_record).collect(),
                Some(PcSaftBinaryRecord::new(Some(kij[0]), None, epsilon_k_ab)),
            )
            .unwrap();
            let eos = Arc::new(PcSaft::new(Arc::new(params)));
            PhaseEquilibrium::bubble_point(
                &eos,
                t * KELVIN,
                &arr1(&[x, 1.0 - x]),
                Some(p * PASCAL),
                None,
                Default::default(),
            )
            .ok()
        });
    filter_binary(vles)
}

fn dew_point_(
    parameters: ArrayView3<f64>,
    kij: ArrayView2<f64>,
    temperature: ArrayView1<f64>,
    vapor_molefracs: ArrayView1<f64>,
    pressure: ArrayView1<f64>,
) -> (Array2<f64>, Array1<bool>) {
    let vles = Zip::from(parameters.outer_iter())
        .and(kij.outer_iter())
        .and(temperature)
        .and(vapor_molefracs)
        .and(pressure)
        .par_map_collect(|par, kij, &t, &y, &p| {
            let epsilon_k_ab = if kij[1] == 0.0 { None } else { Some(kij[1]) };
            let params = PcSaftParameters::new_binary(
                par.outer_iter().map(build_record).collect(),
                Some(PcSaftBinaryRecord::new(Some(kij[0]), None, epsilon_k_ab)),
            )
            .unwrap();
            let eos = Arc::new(PcSaft::new(Arc::new(params)));
            PhaseEquilibrium::dew_point(
                &eos,
                t * KELVIN,
                &arr1(&[y, 1.0 - y]),
                Some(p * PASCAL),
                None,
                Default::default(),
            )
            .ok()
        });
    filter_binary(vles)
}

fn filter_binary<E>(vles: Array1<Option<PhaseEquilibrium<E, 2>>>) -> (Array2<f64>, Array1<bool>) {
    let status = vles.iter().map(|v| v.is_none()).collect();
    let vles: Array1<_> = vles.into_iter().flatten().collect();
    let mut rho = Array2::zeros([vles.len(), 4]);
    Zip::from(rho.rows_mut())
        .and(&vles)
        .for_each(|mut rho, vle| {
            let rho_v = vle.vapor().partial_density.to_reduced();
            let rho_l = vle.liquid().partial_density.to_reduced();
            rho.slice_mut(s![0..2usize]).assign(&rho_v);
            rho.slice_mut(s![2..4usize]).assign(&rho_l);
        });
    (rho, status)
}
