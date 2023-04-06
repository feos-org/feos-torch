#![warn(clippy::all)]
#![allow(clippy::borrow_deref_ref)]
use feos::association::AssociationRecord;
use feos::gc_pcsaft::{GcPcSaft, GcPcSaftEosParameters, GcPcSaftRecord};
use feos_core::joback::JobackRecord;
use feos_core::parameter::{BinaryRecord, ChemicalRecord, ParameterHetero, SegmentRecord};
use feos_core::PhaseEquilibrium;
use ndarray::{arr1, s, Array2, ArrayView1, Zip};
use numpy::{PyArray2, PyReadonlyArray1, ToPyArray};
use pyo3::prelude::*;
use quantity::si::{ANGSTROM, KELVIN, NAV, PASCAL};
use std::sync::Arc;

#[pyclass]
pub struct GcPcSaftParallel {
    segment_records: Vec<SegmentRecord<GcPcSaftRecord, JobackRecord>>,
    chemical_records: Vec<[ChemicalRecord; 2]>,
    binary_segment_records: Vec<BinaryRecord<String, f64>>,
    phi: Array2<f64>,
}

#[pymethods]
impl GcPcSaftParallel {
    #[new]
    fn new(
        segment_records: Vec<(String, PyReadonlyArray1<f64>)>,
        segments: Vec<[Vec<String>; 2]>,
        bonds: Vec<[Vec<[usize; 2]>; 2]>,
        binary_segment_records: Vec<(String, String, f64)>,
        phi: &PyArray2<f64>,
    ) -> Self {
        let segment_records = segment_records
            .into_iter()
            .map(|(i, m)| {
                let m = m.as_array();
                SegmentRecord::new(
                    i,
                    0.0,
                    GcPcSaftRecord::new(
                        m[0],
                        m[1],
                        m[2],
                        Some(m[3]),
                        Some(AssociationRecord::new(m[4], m[5], m[6], m[7], 0.0)),
                        None,
                    ),
                    None,
                )
            })
            .collect();
        Self {
            segment_records,
            chemical_records: segments
                .into_iter()
                .zip(bonds.into_iter())
                .map(|([s1, s2], [b1, b2])| {
                    [
                        ChemicalRecord::new(Default::default(), s1, Some(b1)),
                        ChemicalRecord::new(Default::default(), s2, Some(b2)),
                    ]
                })
                .collect(),
            binary_segment_records: binary_segment_records
                .into_iter()
                .map(|(s1, s2, k)| BinaryRecord::new(s1, s2, k))
                .collect(),
            phi: phi.to_owned_array(),
        }
    }

    fn bubble_point<'py>(
        &self,
        temperature: PyReadonlyArray1<f64>,
        liquid_molefracs: PyReadonlyArray1<f64>,
        pressure: PyReadonlyArray1<f64>,
        py: Python<'py>,
    ) -> &'py PyArray2<f64> {
        self.bubble_point_(
            temperature.as_array(),
            liquid_molefracs.as_array(),
            pressure.as_array(),
        )
        .view()
        .to_pyarray(py)
    }

    fn dew_point<'py>(
        &self,
        temperature: PyReadonlyArray1<f64>,
        vapor_molefracs: PyReadonlyArray1<f64>,
        pressure: PyReadonlyArray1<f64>,
        py: Python<'py>,
    ) -> &'py PyArray2<f64> {
        self.dew_point_(
            temperature.as_array(),
            vapor_molefracs.as_array(),
            pressure.as_array(),
        )
        .view()
        .to_pyarray(py)
    }
}

impl GcPcSaftParallel {
    fn bubble_point_(
        &self,
        temperature: ArrayView1<f64>,
        liquid_molefracs: ArrayView1<f64>,
        pressure: ArrayView1<f64>,
    ) -> Array2<f64> {
        let mut rho = Array2::zeros([temperature.len(), 4]);
        Zip::from(rho.rows_mut())
            .and(ArrayView1::from(&self.chemical_records))
            .and(temperature)
            .and(liquid_molefracs)
            .and(pressure)
            .and(self.phi.rows())
            .par_for_each(|mut rho, cr, &t, &x, &p, phi| {
                let [cr1, cr2] = cr.clone();
                let params = GcPcSaftEosParameters::from_segments(
                    vec![cr1, cr2],
                    self.segment_records.clone(),
                    Some(self.binary_segment_records.clone()),
                )
                .unwrap()
                .phi(phi.as_slice().unwrap())
                .unwrap();
                let eos = Arc::new(GcPcSaft::new(Arc::new(params)));
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
                        let rho_v = vle
                            .vapor()
                            .partial_density
                            .to_reduced(ANGSTROM.powi(-3) / NAV)
                            .unwrap();
                        let rho_l = vle
                            .liquid()
                            .partial_density
                            .to_reduced(ANGSTROM.powi(-3) / NAV)
                            .unwrap();
                        rho.slice_mut(s![0..2usize]).assign(&rho_v);
                        rho.slice_mut(s![2..4usize]).assign(&rho_l);
                    }
                }
            });
        rho
    }

    fn dew_point_(
        &self,
        temperature: ArrayView1<f64>,
        vapor_molefracs: ArrayView1<f64>,
        pressure: ArrayView1<f64>,
    ) -> Array2<f64> {
        let mut rho = Array2::zeros([temperature.len(), 4]);
        Zip::from(rho.rows_mut())
            .and(ArrayView1::from(&self.chemical_records))
            .and(temperature)
            .and(vapor_molefracs)
            .and(pressure)
            .and(self.phi.rows())
            .par_for_each(|mut rho, cr, &t, &y, &p, phi| {
                let [cr1, cr2] = cr.clone();
                let params = GcPcSaftEosParameters::from_segments(
                    vec![cr1, cr2],
                    self.segment_records.clone(),
                    Some(self.binary_segment_records.clone()),
                )
                .unwrap()
                .phi(phi.as_slice().unwrap())
                .unwrap();
                let eos = Arc::new(GcPcSaft::new(Arc::new(params)));
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
                        let rho_v = vle
                            .vapor()
                            .partial_density
                            .to_reduced(ANGSTROM.powi(-3) / NAV)
                            .unwrap();
                        let rho_l = vle
                            .liquid()
                            .partial_density
                            .to_reduced(ANGSTROM.powi(-3) / NAV)
                            .unwrap();
                        rho.slice_mut(s![0..2usize]).assign(&rho_v);
                        rho.slice_mut(s![2..4usize]).assign(&rho_l);
                    }
                }
            });
        rho
    }
}
