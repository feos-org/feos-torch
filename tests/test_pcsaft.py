import torch
import numpy as np
from feos_torch.pcsaft import PcSaftPure
from feos.eos import State, Contributions, PhaseEquilibrium, EquationOfState
from feos.pcsaft import PcSaftParameters, PcSaftRecord, Identifier, PureRecord
from feos.si import ANGSTROM, NAV, KELVIN, KB, PASCAL, KILO, MOL, METER


def pcsaft(params):
    torch.autograd.set_detect_anomaly(True)
    temperature = torch.tensor([300], dtype=torch.float64)
    density = torch.tensor([0.001], dtype=torch.float64)
    temperature_si = temperature.item() * KELVIN
    density_si = density.item() / NAV / ANGSTROM**3

    record = PcSaftRecord(
        params[0],
        params[1],
        params[2],
        params[3],
        kappa_ab=params[4],
        epsilon_k_ab=params[5],
    )
    record = PureRecord(Identifier(), 1, record)
    pcsaft = EquationOfState.pcsaft(PcSaftParameters.new_pure(record))

    x = torch.tensor([params], requires_grad=True, dtype=torch.float64)
    eos = PcSaftPure(x)
    a, p, dp = eos.derivatives(temperature, density)

    s = State(pcsaft, temperature_si, density=density_si)
    a_feos = (
        s.helmholtz_energy(Contributions.ResidualNvt)
        / s.volume
        / (KB * temperature * KELVIN)
        * ANGSTROM**3
    )
    p_feos = s.pressure() / (KB * temperature * KELVIN) * ANGSTROM**3
    dp_feos = s.dp_drho() * s.total_moles / (KB * temperature * KELVIN)
    print("Helmholtz energy derivatives")
    print(f"python: {a.item():.16f} {p.item():.16f} {dp.item():.16f}")
    print(f"feos:   {a_feos:.16f} {p_feos:.16f} {dp_feos:.16f}\n")
    assert np.abs(a.item() - a_feos) < 1e-10
    assert np.abs(p.item() - p_feos) < 1e-10
    assert np.abs(dp.item() - dp_feos) < 1e-10

    p_vap = eos.vapor_pressure(temperature)
    p_vap_feos = PhaseEquilibrium.pure(pcsaft, temperature_si).vapor.pressure() / PASCAL
    print("Vapor pressure")
    print(f"python: {p_vap.item():.16f}")
    print(f"feos:   {p_vap_feos:.16f}\n")
    assert np.abs(p_vap.item() - p_vap_feos) / p_vap_feos < 1e-10

    pressure = torch.tensor([1e5], dtype=torch.float64)
    rho_liq = eos.liquid_density(temperature, pressure)
    rho_liq_feos = State(
        pcsaft,
        temperature_si,
        pressure=pressure.item() * PASCAL,
        density_initialization="liquid",
    ).density / (KILO * MOL / METER**3)
    print("Liquid density")
    print(f"python: {rho_liq.item():.16f}")
    print(f"feos:   {rho_liq_feos:.16f}\n")
    assert np.abs(rho_liq.item() - rho_liq_feos) / rho_liq_feos < 1e-10


def test_pcsaft_full():
    pcsaft([1.5, 3.2, 150, 2.5, 0.03, 2500])


def test_pcsaft_non_assoc():
    pcsaft([1.5, 3.2, 150, 2.5, 0, 2500])
    pcsaft([1.5, 3.2, 150, 2.5, 0.03, 0])
    pcsaft([1.5, 3.2, 150, 2.5, 0, 0])


def test_pcsaft_non_polar():
    pcsaft([1.5, 3.2, 150, 0, 0.03, 2500])


def test_gradients():
    params = [1.5, 3.2, 150, 2.5, 0.03, 2500]
    temperature = torch.tensor([300], dtype=torch.float64)
    pressure = torch.tensor([1e5], dtype=torch.float64)

    x = torch.tensor([params], requires_grad=True, dtype=torch.float64)
    eos = PcSaftPure(x)
    eos.liquid_density(temperature, pressure).backward()

    h = 0.000000005
    rho0 = PcSaftPure(x).liquid_density(temperature, pressure)
    print(x.grad)
    for i in range(6):
        hi = params[i] * h
        xh = [xj + hi if j == i else xj for j, xj in enumerate(params)]
        xh = torch.tensor([xh], requires_grad=True, dtype=torch.float64)
        grad = (PcSaftPure(xh).liquid_density(temperature, pressure) - rho0) / hi
        print(np.abs((grad.item() - x.grad[0, i].item()) / x.grad[0, i].item()))
        assert np.abs((grad.item() - x.grad[0, i].item()) / x.grad[0, i].item()) < 1e-4
