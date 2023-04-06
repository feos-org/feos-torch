import torch
import numpy as np
from feos_torch.pcsaft_pure import PcSaftPure
from feos.eos import State, Contributions, PhaseEquilibrium, EquationOfState
from feos.pcsaft import PcSaftParameters, PcSaftRecord, Identifier, PureRecord
from feos.si import ANGSTROM, NAV, KELVIN, KB, PASCAL, KILO, MOL, METER


def test_pcsaft():
    params = [
        [1.5, 3.2, 350, 0, 0, 0, 0, 0],
        [1.5, 3.2, 150, 2.5, 0.03, 2500, 2, 1],
        [1.5, 3.2, 150, 2.5, 0, 2500, 1, 1],
        [1.5, 3.2, 150, 2.5, 0.03, 0, 1, 1],
        [1.5, 3.2, 150, 2.5, 0, 0, 0, 0],
        [1.5, 3.2, 150, 2.5, 0.03, 2500, 0, 2],
    ]

    temperature = torch.tensor([300] * len(params), dtype=torch.float64)
    pressure = torch.tensor([1e5] * len(params), dtype=torch.float64)
    density = torch.tensor([0.001] * len(params), dtype=torch.float64)
    temperature_si = temperature[0].item() * KELVIN
    pressure_si = pressure[0].item() * PASCAL
    density_si = density[0].item() / NAV / ANGSTROM**3

    x = torch.tensor(params, requires_grad=True, dtype=torch.float64)
    eos = PcSaftPure(x)
    a, p, dp = eos.derivatives(temperature, density)
    _, p_vap = eos.vapor_pressure(temperature)
    _, rho_liq = eos.liquid_density(temperature, pressure)
    _, rho_liq_eq = eos.equilibrium_liquid_density(temperature)

    for i, param in enumerate(params):
        record = PcSaftRecord(
            param[0],
            param[1],
            param[2],
            param[3],
            kappa_ab=param[4],
            epsilon_k_ab=param[5],
            na=param[6],
            nb=param[7],
        )
        record = PureRecord(Identifier(), 1, record)
        pcsaft = EquationOfState.pcsaft(PcSaftParameters.new_pure(record))

        s = State(pcsaft, temperature_si, density=density_si)
        a_feos = (
            s.helmholtz_energy(Contributions.ResidualNvt)
            / s.volume
            / (KB * temperature_si)
            * ANGSTROM**3
        )
        p_feos = s.pressure() / (KB * temperature_si) * ANGSTROM**3
        dp_feos = s.dp_drho() * s.total_moles / (KB * temperature_si)
        print("Helmholtz energy derivatives")
        print(f"python: {a[i].item():.16f} {p[i].item():.16f} {dp[i].item():.16f}")
        print(f"feos:   {a_feos:.16f} {p_feos:.16f} {dp_feos:.16f}\n")
        assert np.abs(a[i].item() - a_feos) < 1e-10
        assert np.abs(p[i].item() - p_feos) < 1e-10
        assert np.abs(dp[i].item() - dp_feos) < 1e-10

        p_vap_feos = (
            PhaseEquilibrium.pure(pcsaft, temperature_si).vapor.pressure() / PASCAL
        )
        print("Vapor pressure")
        print(f"python: {p_vap[i].item():.16f}")
        print(f"feos:   {p_vap_feos:.16f}\n")
        assert np.abs(p_vap[i].item() - p_vap_feos) / p_vap_feos < 1e-10

        rho_liq_feos = State(
            pcsaft,
            temperature_si,
            pressure=pressure_si,
            density_initialization="liquid",
        ).density / (KILO * MOL / METER**3)
        print("Liquid density")
        print(f"python: {rho_liq[i].item():.16f}")
        print(f"feos:   {rho_liq_feos:.16f}\n")
        assert np.abs(rho_liq[i].item() - rho_liq_feos) / rho_liq_feos < 1e-10

        rho_liq_eq_feos = PhaseEquilibrium.pure(
            pcsaft, temperature_si
        ).liquid.density / (KILO * MOL / METER**3)
        print("Equilibrium liquid density")
        print(f"python: {rho_liq_eq[i].item():.16f}")
        print(f"feos:   {rho_liq_eq_feos:.16f}\n")
        assert np.abs(rho_liq_eq[i].item() - rho_liq_eq_feos) / rho_liq_eq_feos < 1e-10


def test_gradients_liquid_density():
    params = [1.5, 3.2, 150, 2.5, 0.03, 2500, 1, 1]
    temperature = torch.tensor([300], dtype=torch.float64)
    pressure = torch.tensor([1e5], dtype=torch.float64)

    x = torch.tensor([params], requires_grad=True, dtype=torch.float64)
    eos = PcSaftPure(x)
    eos.liquid_density(temperature, pressure)[1].backward()

    h = 0.000000005
    rho0 = PcSaftPure(x).liquid_density(temperature, pressure)[1]
    print(x.grad)
    for i in range(6):
        hi = params[i] * h
        xh = [xj + hi if j == i else xj for j, xj in enumerate(params)]
        xh = torch.tensor([xh], requires_grad=True, dtype=torch.float64)
        grad = (PcSaftPure(xh).liquid_density(temperature, pressure)[1] - rho0) / hi
        print(
            grad.item(),
            x.grad[0, i].item(),
            np.abs((grad.item() - x.grad[0, i].item()) / x.grad[0, i].item()),
        )
        assert np.abs((grad.item() - x.grad[0, i].item()) / x.grad[0, i].item()) < 1e-4


def test_gradients_vapor_pressure():
    params = [1.5, 3.2, 150, 2.5, 0.03, 2500, 1, 2]
    temperature = torch.tensor([300], dtype=torch.float64)

    x = torch.tensor([params], requires_grad=True, dtype=torch.float64)
    eos = PcSaftPure(x)
    eos.vapor_pressure(temperature)[1].backward()

    h = 0.000000005
    rho0 = PcSaftPure(x).vapor_pressure(temperature)[1]
    print(x.grad)
    for i in range(6):
        hi = params[i] * h
        xh = [xj + hi if j == i else xj for j, xj in enumerate(params)]
        xh = torch.tensor([xh], requires_grad=True, dtype=torch.float64)
        grad = (PcSaftPure(xh).vapor_pressure(temperature)[1] - rho0) / hi
        print(
            grad.item(),
            x.grad[0, i].item(),
            np.abs((grad.item() - x.grad[0, i].item()) / x.grad[0, i].item()),
        )
        assert np.abs((grad.item() - x.grad[0, i].item()) / x.grad[0, i].item()) < 1e-4


def test_gradients_equilibrium_liquid_density():
    params = [1.5, 3.2, 150, 2.5, 0.03, 2500, 2, 1]
    temperature = torch.tensor([300], dtype=torch.float64)

    x = torch.tensor([params], requires_grad=True, dtype=torch.float64)
    eos = PcSaftPure(x)
    eos.equilibrium_liquid_density(temperature)[1].backward()

    h = 0.000000005
    rho0 = PcSaftPure(x).equilibrium_liquid_density(temperature)[1]
    print(x.grad)
    for i in range(6):
        hi = params[i] * h
        xh = [xj + hi if j == i else xj for j, xj in enumerate(params)]
        xh = torch.tensor([xh], requires_grad=True, dtype=torch.float64)
        grad = (PcSaftPure(xh).equilibrium_liquid_density(temperature)[1] - rho0) / hi
        print(
            grad.item(),
            x.grad[0, i].item(),
            np.abs((grad.item() - x.grad[0, i].item()) / x.grad[0, i].item()),
        )
        assert np.abs((grad.item() - x.grad[0, i].item()) / x.grad[0, i].item()) < 1e-4
