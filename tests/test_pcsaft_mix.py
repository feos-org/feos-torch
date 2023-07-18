import torch
import numpy as np
from feos_torch.pcsaft_mix import PcSaftMix
from feos.eos import State, Contributions, PhaseEquilibrium, EquationOfState
from feos.pcsaft import (
    PcSaftParameters,
    PcSaftRecord,
    Identifier,
    PureRecord,
    PcSaftBinaryRecord,
    BinaryRecord,
)
from feos.si import ANGSTROM, NAV, KELVIN, KB, PASCAL, RGAS, BAR


def test_pcsaft():
    params = [
        [[1.5, 3.2, 150, 0, 0, 0, 0, 0], [2.5, 3.5, 250, 0, 0, 0, 0, 0]],
        [[1.5, 3.2, 150, 2.5, 0, 0, 0, 0], [2.5, 3.5, 250, 0, 0, 0, 0, 0]],
        [[1.5, 3.2, 150, 0, 0, 0, 0, 0], [2.5, 3.5, 250, 2, 0, 0, 0, 0]],
        [[1.5, 3.2, 150, 2.5, 0, 0, 0, 0], [2.5, 3.5, 250, 2, 0, 0, 0, 0]],
        [[1.5, 3.2, 150, 0, 0.03, 2500, 2, 1], [2.5, 3.5, 250, 0, 0, 0, 0, 0]],
        [[1.5, 3.2, 150, 0, 0, 0, 0, 0], [2.5, 3.5, 250, 0, 0.025, 1500, 1, 2]],
        [[1.5, 3.2, 150, 0, 0.03, 2500, 1, 1], [2.5, 3.5, 250, 0, 0.025, 1500, 1, 1]],
        [[1.5, 3.2, 150, 2.5, 0.03, 2500, 1, 1], [2.5, 3.5, 250, 2, 0.025, 1500, 1, 1]],
        [[1.5, 3.2, 150, 0, 0.03, 2500, 1, 1], [2.5, 3.5, 250, 0, 0.025, 1500, 0, 1]],
        [[1.5, 3.2, 150, 0, 0.03, -500, 0, 2], [2.5, 3.5, 250, 0, 0.025, 1500, 1, 1]],
        [[1.5, 3.2, 150, 0, 0, 0, 0, 0], [2.5, 3.5, 250, 0, 0.025, 1500, 0, 1]],
        [[1.5, 3.2, 150, 0, 0.03, 2500, 2, 2], [2.5, 3.5, 250, 0, 0.025, 1500, 1, 1]],
        [[1.5, 3.2, 150, 0, 0.03, 2500, 2, 2], [2.5, 3.5, 250, 0, 0.025, 1500, 1, 1]],
        [[1.5, 3.2, 150, 0, 0.03, 2500, 1, 2], [2.5, 3.5, 250, 0, 0.025, 1500, 2, 1]],
    ]
    kij = torch.tensor([[-0.05, 0]] * len(params), dtype=torch.float64)
    kij[12, 1] = 3000
    x = torch.tensor(params, dtype=torch.float64)
    T = 300
    temperature = torch.tensor([T] * len(params), dtype=torch.float64)
    rho = [0.001, 0.002]
    density = torch.tensor([rho] * len(params), dtype=torch.float64)

    records = [
        [
            PcSaftRecord(
                p[0],
                p[1],
                p[2],
                p[3],
                kappa_ab=p[4],
                epsilon_k_ab=p[5],
                na=p[6],
                nb=p[7],
            )
            for p in param
        ]
        for param in params
    ]
    records = [[PureRecord(Identifier(), 1, r) for r in record] for record in records]
    pcsaft = [
        EquationOfState.pcsaft(
            PcSaftParameters.new_binary(
                record,
                kij
                if epsilon_k_ab == 0
                else PcSaftBinaryRecord(kij, None, epsilon_k_ab),
            )
        )
        for record, (kij, epsilon_k_ab) in zip(records, kij)
    ]
    states = [
        State(eos, T * KELVIN, partial_density=np.array(rho) / (NAV * ANGSTROM**3))
        for eos in pcsaft
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

    eos = PcSaftMix(x, kij)
    a = eos.helmholtz_energy_density(temperature, density)
    _, p, mu, v = eos.derivatives(temperature, density)

    for i, s in enumerate(
        [
            "np/np",
            "p/np",
            "np/p",
            "p/p",
            "a/np",
            "np/a",
            "a/a",
            "ap/ap",
            "a/x",
            "x/a",
            "np/x",
            "aa/a",
            "a/a k",
            "aa/aa",
        ]
    ):
        print(
            f"{s:5} feos: {a_feos[i]:.16f} {mu1_feos[i]:.16f} {mu2_feos[i]:.16f} {p_feos[i]:.16f} {v1_feos[i]:.16f} {v2_feos[i]:.16f}"
        )
        print(
            f"     torch: {a[i].item():.16f} {mu[i,0].item():.16f} {mu[i,1].item():.16f} {p[i].item():.16f} {v[i,0].item():.16f} {v[i,1].item():.16f}\n"
        )

        assert np.abs(a_feos[i] - a[i].item()) < 1e-14
        assert np.abs(mu1_feos[i] - mu[i, 0].item()) < 1e-14
        assert np.abs(mu2_feos[i] - mu[i, 1].item()) < 1e-14
        assert np.abs(p_feos[i] - p[i].item()) < 1e-14
        assert np.abs(v1_feos[i] - v[i, 0].item()) < 1e-11
        assert np.abs(v2_feos[i] - v[i, 1].item()) < 1e-11


def test_bubble_point():
    h = 1e-8
    kij = -0.15
    epsilon_k_aibj = 1000
    kij = torch.tensor(
        [[kij, epsilon_k_aibj], [kij + h, epsilon_k_aibj]],
        dtype=torch.float64,
        requires_grad=True,
    )
    params = torch.tensor(
        [[[1, 3.5, 150, 0, 0.02, 1500, 1, 1], [1, 3.5, 200, 0, 0.03, 2500, 1, 1]]]
        * len(kij),
        dtype=torch.float64,
    )
    temperature = torch.tensor(
        [150] * len(params), dtype=torch.float64, requires_grad=True
    )
    pressure = torch.tensor(
        [1e5] * len(params), dtype=torch.float64, requires_grad=True
    )
    liquid_molefracs = torch.tensor(
        [0.5] * len(params), dtype=torch.float64, requires_grad=True
    )
    eos = PcSaftMix(params, kij)
    p, _ = eos.bubble_point(temperature, liquid_molefracs, pressure)
    p[0].backward()
    print(kij.grad[(0, 0)].item())

    records = [
        [
            PcSaftRecord(
                p[0],
                p[1],
                p[2],
                p[3],
                kappa_ab=p[4],
                epsilon_k_ab=p[5],
                na=p[6],
                nb=p[7],
            )
            for p in param
        ]
        for param in params
    ]
    records = [[PureRecord(Identifier(), 1, r) for r in record] for record in records]
    pcsaft = [
        EquationOfState.pcsaft(
            PcSaftParameters.new_binary(
                record,
                PcSaftBinaryRecord(kij, None, epsilon_k_aibj),
            )
        )
        for record, (kij, epsilon_k_aibj) in zip(records, kij)
    ]
    p_feos = [
        PhaseEquilibrium.bubble_point(
            eos, 150 * KELVIN, np.array([0.5, 0.5]), BAR
        ).vapor.pressure()
        / PASCAL
        for eos in pcsaft
    ]
    print((p_feos[1] - p_feos[0]) / h)

    assert np.abs(p[0].item() - p_feos[0]) < 1e-8
    assert np.abs(p[1].item() - p_feos[1]) < 1e-8
    assert np.abs(kij.grad[(0, 0)].item() - (p_feos[1] - p_feos[0]) / h) < 1


def test_dew_point():
    h = 1e-8
    kij = -0.15
    kij = torch.tensor(
        [[kij, 0], [kij + h, 0]], dtype=torch.float64, requires_grad=True
    )
    params = torch.tensor(
        [[[1, 3.5, 150, 0, 0, 0, 0, 0], [1, 3.5, 200, 0, 0, 0, 0, 0]]] * len(kij),
        dtype=torch.float64,
    )
    temperature = torch.tensor(
        [150] * len(params), dtype=torch.float64, requires_grad=True
    )
    pressure = torch.tensor(
        [1e5] * len(params), dtype=torch.float64, requires_grad=True
    )
    vapor_molefracs = torch.tensor(
        [0.5] * len(params), dtype=torch.float64, requires_grad=True
    )
    eos = PcSaftMix(params, kij)
    p, _ = eos.dew_point(temperature, vapor_molefracs, pressure)
    p[0].backward()
    print(kij.grad[(0, 0)].item())

    records = [
        [
            PcSaftRecord(
                p[0],
                p[1],
                p[2],
                p[3],
                kappa_ab=p[4],
                epsilon_k_ab=p[5],
            )
            for p in param
        ]
        for param in params
    ]
    records = [[PureRecord(Identifier(), 1, r) for r in record] for record in records]
    pcsaft = [
        EquationOfState.pcsaft(PcSaftParameters.new_binary(record, kij))
        for record, (kij, _) in zip(records, kij)
    ]
    p_feos = [
        PhaseEquilibrium.dew_point(
            eos, 150 * KELVIN, np.array([0.5, 0.5]), BAR
        ).vapor.pressure()
        / PASCAL
        for eos in pcsaft
    ]
    print((p_feos[1] - p_feos[0]) / h)
    print(p)
    print(p_feos)

    assert np.abs(p[0].item() - p_feos[0]) < 1e-8
    assert np.abs(p[1].item() - p_feos[1]) < 1e-8
    assert np.abs(kij.grad[(0, 0)].item() - (p_feos[1] - p_feos[0]) / h) < 1
