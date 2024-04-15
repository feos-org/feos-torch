import torch
import numpy as np

from si_units import KELVIN, KB, ANGSTROM, NAV, PASCAL, MOL, METER, KILO, JOULE
from feos_torch import PcSaft


from .dual import Dual3

A0 = [
    0.91056314451539,
    0.63612814494991,
    2.68613478913903,
    -26.5473624914884,
    97.7592087835073,
    -159.591540865600,
    91.2977740839123,
]
A1 = [
    -0.30840169182720,
    0.18605311591713,
    -2.50300472586548,
    21.4197936296668,
    -65.2558853303492,
    83.3186804808856,
    -33.7469229297323,
]
A2 = [
    -0.09061483509767,
    0.45278428063920,
    0.59627007280101,
    -1.72418291311787,
    -4.13021125311661,
    13.7766318697211,
    -8.67284703679646,
]
B0 = [
    0.72409469413165,
    2.23827918609380,
    -4.00258494846342,
    -21.00357681484648,
    26.8556413626615,
    206.5513384066188,
    -355.60235612207947,
]
B1 = [
    -0.57554980753450,
    0.69950955214436,
    3.89256733895307,
    -17.21547164777212,
    192.6722644652495,
    -161.8264616487648,
    -165.2076934555607,
]
B2 = [
    0.09768831158356,
    -0.25575749816100,
    -9.15585615297321,
    20.64207597439724,
    -38.80443005206285,
    93.6267740770146,
    -29.66690558514725,
]

AD = [
    [0.30435038064, 0.95346405973, -1.16100802773],
    [-0.13585877707, -1.83963831920, 4.52586067320],
    [1.44933285154, 2.01311801180, 0.97512223853],
    [0.35569769252, -7.37249576667, -12.2810377713],
    [-2.06533084541, 8.23741345333, 5.93975747420],
]

BD = [
    [0.21879385627, -0.58731641193, 3.48695755800],
    [-1.18964307357, 1.24891317047, -14.9159739347],
    [1.16268885692, -0.50852797392, 15.3720218600],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
]

CD = [
    [-0.06467735252, -0.95208758351, -0.62609792333],
    [0.19758818347, 2.99242575222, 1.29246858189],
    [-0.80875619458, -2.38026356489, 1.65427830900],
    [0.69028490492, -0.27012609786, -3.43967436378],
]


class PcSaftPure:
    def __init__(self, parameters):
        self.m = parameters[:, 0]
        self.sigma = parameters[:, 1]
        self.epsilon_k = parameters[:, 2]
        self.mu2 = (
            parameters[:, 3] ** 2
            / (self.m * self.sigma**3 * self.epsilon_k)
            * 1e-19
            * (JOULE / KELVIN / KB)
        )
        self.kappa_ab = parameters[:, 4]
        self.epsilon_k_ab = parameters[:, 5]
        self.na = parameters[:, 6]
        self.nb = parameters[:, 7]
        self.parameters = parameters.detach().cpu().numpy()

    def helmholtz_energy(self, temperature, density):
        # temperature dependent segment diameter
        d = self.sigma * (1 - 0.12 * (-3 * self.epsilon_k / temperature).exp())

        eta = np.pi / 6 * self.m * density * d**3
        eta2 = eta * eta
        eta3 = eta2 * eta
        eta_m1 = 1 / (1 - eta)
        eta_m2 = eta_m1 * eta_m1
        etas = [1, eta, eta2, eta3, eta2 * eta2, eta2 * eta3, eta3 * eta3]

        # hard sphere
        hs = self.m * density * (4 * eta - 3 * eta2) * eta_m2

        # hard chain
        g = (1 - eta / 2) * eta_m1 * eta_m2
        hc = -density * (self.m - 1) * g.log()

        # dispersion
        e = self.epsilon_k / temperature
        s3 = self.sigma**3
        I1 = 0
        I2 = 0
        m1 = (self.m - 1) / self.m
        m2 = (self.m - 2) / self.m
        for i in range(7):
            I1 = I1 + (m1 * (m2 * A2[i] + A1[i]) + A0[i]) * etas[i]
            I2 = I2 + (m1 * (m2 * B2[i] + B1[i]) + B0[i]) * etas[i]
        C1 = 1 / (
            1
            + self.m * (8 * eta - 2 * eta2) * eta_m2 * eta_m2
            + (1 - self.m)
            * (20 * eta - 27 * eta2 + 12 * eta2 * eta - 2 * eta2 * eta2)
            / ((1 - eta) * (1 - eta) * (2 - eta) * (2 - eta))
        )
        I = 2 * I1 + C1 * I2 * self.m * e
        disp = (-np.pi * density * density * self.m**2 * e * s3) * I

        # dipoles
        mu2 = self.mu2 * e * s3
        m = self.m.clamp(max=2)
        m1 = (m - 1) / m
        m2 = m1 * (m - 2) / m
        J1 = 0
        for i in range(5):
            a = AD[i][0] + m1 * AD[i][1] + m2 * AD[i][2]
            b = BD[i][0] + m1 * BD[i][1] + m2 * BD[i][2]
            J1 = J1 + (a + b * e) * etas[i]
        J2 = sum((CD[i][0] + m1 * CD[i][1] + m2 * CD[i][2]) * etas[i] for i in range(4))

        PI_SQ_43 = 4 / 3 * np.pi**2
        # mu is factored out of these expressions to deal with the case where mu=0
        phi2 = -density * density * J1 / s3 * np.pi
        phi3 = -density * density * density * J2 / s3 * PI_SQ_43
        dipole = phi2 * phi2 * mu2 * mu2 / (phi2 - phi3 * mu2)

        # association
        delta_assoc = (
            ((self.epsilon_k_ab / temperature).exp() - 1)
            * self.sigma**3
            * self.kappa_ab
        )
        k = eta * eta_m1
        delta = (1 + k * (1.5 + 0.5 * k)) * eta_m1 * delta_assoc
        rhoa = self.na * density
        rhob = self.nb * density
        aux = 1 + (rhoa - rhob) * delta
        sqrt = (aux * aux + 4 * rhob * delta).sqrt()
        xa = 2 / (sqrt + 1 + (rhob - rhoa) * delta)
        xb = 2 / (sqrt + 1 - (rhob - rhoa) * delta)
        assoc = rhoa * (xa.log() - 0.5 * xa + 0.5) + rhob * (xb.log() - 0.5 * xb + 0.5)

        return hs + hc + disp + dipole + assoc

    def derivatives(self, temperature, density):
        a = self.helmholtz_energy(temperature, Dual3.diff(density))
        return a.re, density - a.re + density * a.v1, 1 + density * a.v2

    def liquid_density(self, temperature, pressure):
        density, nans = PcSaft.liquid_density(
            self.parameters,
            temperature.detach().cpu().numpy(),
            pressure.detach().cpu().numpy(),
        )
        density = torch.from_numpy(density).to(self.m.device)
        nans = torch.from_numpy(nans).to(self.m.device)
        temperature = temperature[~nans]
        pressure = pressure[~nans]
        self.reduce(nans)

        pressure = pressure / temperature * (PASCAL / (KB * KELVIN) * ANGSTROM**3)
        _, p, dp = self.derivatives(temperature, density)
        density = density - (p - pressure) / dp
        return nans, density / ((KILO * MOL / METER**3) * (NAV * ANGSTROM**3))

    def vapor_pressure(self, temperature):
        density, nans = PcSaft.vapor_pressure(
            self.parameters, temperature.detach().cpu().numpy()
        )
        density = torch.from_numpy(density).to(self.m.device)
        nans = torch.from_numpy(nans).to(self.m.device)
        temperature = temperature[~nans]
        self.reduce(nans)

        rho_V = density[:, 0]
        rho_L = density[:, 1]
        a_L = self.helmholtz_energy(temperature, rho_L) / rho_L
        a_V = self.helmholtz_energy(temperature, rho_V) / rho_V
        p = -(a_V - a_L + (rho_V / rho_L).log()) / (1 / rho_V - 1 / rho_L)
        return nans, p * temperature * (KB * KELVIN / ANGSTROM**3 / PASCAL)

    def equilibrium_liquid_density(self, temperature):
        density, nans = PcSaft.vapor_pressure(
            self.parameters, temperature.detach().cpu().numpy()
        )
        density = torch.from_numpy(density).to(self.m.device)
        nans = torch.from_numpy(nans).to(self.m.device)
        temperature = temperature[~nans]
        self.reduce(nans)

        rho_V = density[:, 0]
        rho_L = density[:, 1]
        a_L, p_L, dp_L = self.derivatives(temperature, rho_L)
        a_L /= rho_L
        a_V = self.helmholtz_energy(temperature, rho_V) / rho_V
        p = -(a_V - a_L + (rho_V / rho_L).log()) / (1 / rho_V - 1 / rho_L)
        liquid_density = rho_L - (p_L - p) / dp_L
        return nans, liquid_density / ((KILO * MOL / METER**3) * (NAV * ANGSTROM**3))

    def reduce(self, nans):
        self.m = self.m[~nans]
        self.sigma = self.sigma[~nans]
        self.epsilon_k = self.epsilon_k[~nans]
        self.mu2 = self.mu2[~nans]
        self.kappa_ab = self.kappa_ab[~nans]
        self.epsilon_k_ab = self.epsilon_k_ab[~nans]
        self.na = self.na[~nans]
        self.nb = self.nb[~nans]
