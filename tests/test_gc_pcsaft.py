import torch
import numpy as np
import json
from feos_torch.gc_pcsaft import GcPcSaft
from feos.eos import State, Contributions, PhaseEquilibrium, EquationOfState
from feos.gc_pcsaft import (
    GcPcSaftEosParameters,
    Identifier,
    ChemicalRecord,
    SegmentRecord,
    BinarySegmentRecord,
)
from feos.si import ANGSTROM, NAV, KELVIN, KB, PASCAL, RGAS, BAR


def test_gc_pcsaft():
    segment_lists = [
        [["CH3", "CH2", "CH2", "CH3"], ["CH3", "CH2", "CH3"]],
        [["CH3", ">CH", "CH3", "CH3"], ["CH3", ">C<", "CH3", "CH3", "CH3"]],
        [["CH3", ">CH", "CH3", "CH=O"], ["CH3", ">C<", "CH3", "CH3", "CH3"]],
        [["CH3", ">CH", "CH3", "CH3"], ["CH3", ">C<", "CH3", "CH3", "HCOO"]],
        [["CH3", ">CH", "CH3", "CH=O"], ["CH3", ">C<", "CH3", "CH3", "HCOO"]],
        [["CH3", ">CH", "CH3", "OH"], ["CH3", ">C<", "CH3", "CH3", "CH3"]],
        [["CH3", ">CH", "CH3", "CH3"], ["CH3", ">C<", "CH3", "CH3", "NH2"]],
        [["CH3", ">CH", "CH3", "OH"], ["CH3", ">C<", "CH3", "CH3", "NH2"]],
        [["CH3", ">CH", "CH=O", "OH"], ["CH3", ">C<", "CH3", "HCOO", "NH2"]],
        [["CH3", ">CH", "CH=O", "OH"], ["CH3", ">C<", "CH3", "CH2", "IA"]],
        [["CH3", ">CH", "CH=O", "IA"], ["CH3", ">C<", "CH3", "CH2", "IA"]],
    ]
    bond_lists = [
        [[[0, 1], [1, 2], [2, 3]], [[0, 1], [1, 2]]],
        [[[0, 1], [1, 2], [1, 3]], [[0, 1], [1, 2], [1, 3], [1, 4]]],
        [[[0, 1], [1, 2], [1, 3]], [[0, 1], [1, 2], [1, 3], [1, 4]]],
        [[[0, 1], [1, 2], [1, 3]], [[0, 1], [1, 2], [1, 3], [1, 4]]],
        [[[0, 1], [1, 2], [1, 3]], [[0, 1], [1, 2], [1, 3], [1, 4]]],
        [[[0, 1], [1, 2], [1, 3]], [[0, 1], [1, 2], [1, 3], [1, 4]]],
        [[[0, 1], [1, 2], [1, 3]], [[0, 1], [1, 2], [1, 3], [1, 4]]],
        [[[0, 1], [1, 2], [1, 3]], [[0, 1], [1, 2], [1, 3], [1, 4]]],
        [[[0, 1], [1, 2], [1, 3]], [[0, 1], [1, 2], [1, 3], [1, 4]]],
        [[[0, 1], [1, 2], [1, 3]], [[0, 1], [1, 2], [1, 3], [1, 4]]],
        [[[0, 1], [1, 2], [1, 3]], [[0, 1], [1, 2], [1, 3], [1, 4]]],
    ]
    kab_list = [("CH3", "CH=O", 0.03), (">CH", "HCOO", -0.01)]
    phi = torch.tensor([[1.1, 0.98]] * len(segment_lists), dtype=torch.float64)

    T = 300
    temperature = torch.tensor([T] * len(segment_lists), dtype=torch.float64)
    rho = [0.001, 0.002]
    density = torch.tensor([rho] * len(segment_lists), dtype=torch.float64)

    segment_records = SegmentRecord.from_json("tests/sauer2014_hetero.json")
    chemical_records = [
        [ChemicalRecord(Identifier(), s, b) for s, b in zip(seg, bon)]
        for seg, bon in zip(segment_lists, bond_lists)
    ]
    binary_segment_records = [BinarySegmentRecord(*k) for k in kab_list]
    gc_pcsaft = [
        EquationOfState.gc_pcsaft(
            GcPcSaftEosParameters.from_segments(
                record, segment_records, binary_segment_records
            ).phi(ph.detach().numpy())
        )
        for record, ph in zip(chemical_records, phi)
    ]
    states = [
        State(eos, T * KELVIN, partial_density=np.array(rho) / (NAV * ANGSTROM**3))
        for eos in gc_pcsaft
    ]
    a_feos = [
        s.molar_helmholtz_energy(Contributions.Residual)
        * (s.density / KB / s.temperature * ANGSTROM**3)
        for s in states
    ]
    p_feos = [s.pressure() * (ANGSTROM**3 / KB / s.temperature) for s in states]
    v1_feos = [s.partial_molar_volume()[0] / (NAV * ANGSTROM**3) for s in states]
    v2_feos = [s.partial_molar_volume()[1] / (NAV * ANGSTROM**3) for s in states]
    mu1_feos = [
        s.chemical_potential(Contributions.Residual)[0] / RGAS / s.temperature
        for s in states
    ]
    mu2_feos = [
        s.chemical_potential(Contributions.Residual)[1] / RGAS / s.temperature
        for s in states
    ]

    print(
        [
            (
                n,
                a
                * (1 / states[-1].volume / KB / states[-1].temperature * ANGSTROM**3),
            )
            for n, a in states[-1].residual_helmholtz_energy_contributions()
        ]
    )

    eos = GcPcSaft(*parse_segments(), segment_lists, bond_lists, kab_list, phi)
    a = eos.helmholtz_energy_density(temperature, density)
    _, p, mu, v = eos.derivatives(temperature, density)

    for i, s in enumerate(
        [
            "np/np",
            "np/np (branched)",
            "np/p",
            "p/np",
            "p/p",
            "a/np",
            "np/a",
            "a/a",
            "ap/ap",
            "a/x",
            "x/x",
        ]
    ):
        print(
            f"{s:17} feos: {a_feos[i]:.16f} {mu1_feos[i]:.16f} {mu2_feos[i]:.16f} {p_feos[i]:.16f} {v1_feos[i]:.16f} {v2_feos[i]:.16f}"
        )
        print(
            f"                 torch: {a[i].item():.16f} {mu[i,0].item():.16f} {mu[i,1].item():.16f} {p[i].item():.16f} {v[i,0].item():.16f} {v[i,1].item():.16f}\n"
        )

        assert np.abs(a_feos[i] - a[i].item()) < 1e-14
        assert np.abs(mu1_feos[i] - mu[i, 0].item()) < 1e-14
        assert np.abs(mu2_feos[i] - mu[i, 1].item()) < 1e-14
        assert np.abs(p_feos[i] - p[i].item()) < 1e-14
        assert np.abs(v1_feos[i] - v[i, 0].item()) < 1e-11
        assert np.abs(v2_feos[i] - v[i, 1].item()) < 1e-11


def test_bubble_point():
    h = 1e-8
    kab = torch.tensor([-0.15], dtype=torch.float64, requires_grad=True)
    segment_lists = [
        [["CH3", "CH2", "CH2", "CH3"], ["CH3", "CH2", "CH3"]],
    ]
    bond_lists = [
        [[[0, 1], [1, 2], [2, 3]], [[0, 1], [1, 2]]],
    ]
    kab_list = [(s1, s2, k) for (s1, s2), k in zip([["CH3", "CH2"]], kab)]
    phi = torch.tensor([[1.1, 0.98]], dtype=torch.float64)
    temperature = torch.tensor([150], dtype=torch.float64, requires_grad=True)
    pressure = torch.tensor([1e5], dtype=torch.float64, requires_grad=True)
    liquid_molefracs = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
    eos = GcPcSaft(*parse_segments(), segment_lists, bond_lists, kab_list, phi)
    p, _ = eos.bubble_point(temperature, liquid_molefracs, pressure)
    p[0].backward()
    print(kab.grad[0].item())

    def p_bubble_feos(kab):
        segment_records = SegmentRecord.from_json("tests/sauer2014_hetero.json")
        chemical_records = [
            ChemicalRecord(Identifier(), s, b)
            for s, b in zip(segment_lists[0], bond_lists[0])
        ]
        kab_list = [(s1, s2, k) for (s1, s2), k in zip([["CH3", "CH2"]], kab)]
        binary_segment_records = [BinarySegmentRecord(*k) for k in kab_list]
        gc_pcsaft = EquationOfState.gc_pcsaft(
            GcPcSaftEosParameters.from_segments(
                chemical_records, segment_records, binary_segment_records
            ).phi(phi[0].detach().numpy())
        )
        return (
            PhaseEquilibrium.bubble_point(
                gc_pcsaft, 150 * KELVIN, np.array([0.5, 0.5]), BAR
            ).vapor.pressure()
            / PASCAL
        )

    p_feos = [p_bubble_feos(kab), p_bubble_feos(kab + h)]
    print((p_feos[1] - p_feos[0]) / h)
    print(p_feos)

    assert np.abs(p[0].item() - p_feos[0]) < 1e-8
    assert np.abs(kab.grad[0].item() - (p_feos[1] - p_feos[0]) / h) < 1


def test_dew_point():
    h = 1e-8
    kab = torch.tensor([-0.15], dtype=torch.float64, requires_grad=True)
    segment_lists = [
        [["CH3", "CH2", "CH2", "CH3"], ["CH3", "CH2", "CH3"]],
    ]
    bond_lists = [
        [[[0, 1], [1, 2], [2, 3]], [[0, 1], [1, 2]]],
    ]
    kab_list = [(s1, s2, k) for (s1, s2), k in zip([["CH3", "CH2"]], kab)]
    phi = torch.tensor([[1.1, 0.98]], dtype=torch.float64)
    temperature = torch.tensor([150], dtype=torch.float64, requires_grad=True)
    pressure = torch.tensor([1e5], dtype=torch.float64, requires_grad=True)
    vapor_molefracs = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
    eos = GcPcSaft(*parse_segments(), segment_lists, bond_lists, kab_list, phi)
    p, _ = eos.dew_point(temperature, vapor_molefracs, pressure)
    p[0].backward()
    print(kab.grad[0].item())

    def p_dew_feos(kab):
        segment_records = SegmentRecord.from_json("tests/sauer2014_hetero.json")
        chemical_records = [
            ChemicalRecord(Identifier(), s, b)
            for s, b in zip(segment_lists[0], bond_lists[0])
        ]
        kab_list = [(s1, s2, k) for (s1, s2), k in zip([["CH3", "CH2"]], kab)]
        binary_segment_records = [BinarySegmentRecord(*k) for k in kab_list]
        gc_pcsaft = EquationOfState.gc_pcsaft(
            GcPcSaftEosParameters.from_segments(
                chemical_records, segment_records, binary_segment_records
            ).phi(phi[0].detach().numpy())
        )
        return (
            PhaseEquilibrium.dew_point(
                gc_pcsaft, 150 * KELVIN, np.array([0.5, 0.5]), BAR
            ).vapor.pressure()
            / PASCAL
        )

    p_feos = [p_dew_feos(kab), p_dew_feos(kab + h)]
    print((p_feos[1] - p_feos[0]) / h)
    print(p[0].item())
    print(p_feos)

    assert np.abs(p[0].item() - p_feos[0]) < 1e-8
    assert np.abs(kab.grad[0].item() - (p_feos[1] - p_feos[0]) / h) < 1


def parse_segments():
    with open("tests/sauer2014_hetero.json") as f:
        segment_records = json.load(f)
    segment_identifier = [r["identifier"] for r in segment_records]
    m = torch.tensor(
        [r["model_record"]["m"] for r in segment_records], dtype=torch.float64
    )
    sigma = torch.tensor(
        [r["model_record"]["sigma"] for r in segment_records], dtype=torch.float64
    )
    epsilon_k = torch.tensor(
        [r["model_record"]["epsilon_k"] for r in segment_records], dtype=torch.float64
    )
    mu = torch.tensor(
        [r["model_record"].get("mu", 0) for r in segment_records], dtype=torch.float64
    )
    kappa_ab = torch.tensor(
        [r["model_record"].get("kappa_ab", 0) for r in segment_records],
        dtype=torch.float64,
    )
    epsilon_k_ab = torch.tensor(
        [r["model_record"].get("epsilon_k_ab", 0) for r in segment_records],
        dtype=torch.float64,
    )
    na = torch.tensor(
        [r["model_record"].get("na", 0) for r in segment_records], dtype=torch.float64
    )
    nb = torch.tensor(
        [r["model_record"].get("nb", 0) for r in segment_records], dtype=torch.float64
    )
    parameter = (m, sigma, epsilon_k, mu, kappa_ab, epsilon_k_ab, na, nb)

    return segment_identifier, parameter
