import torch
import numpy as np

from si_units import KELVIN, KB, ANGSTROM, PASCAL, JOULE
from feos_torch import PcSaft


from .dual_torch import Dual2, DualTensor
from .pcsaft_pure import A0, A1, A2, B0, B1, B2, AD, BD, CD


class PcSaftMix:
    def __init__(self, parameters, kij=None):
        self.m = parameters[:, :, 0]
        self.sigma = parameters[:, :, 1]
        self.epsilon_k = parameters[:, :, 2]
        self.mu2 = (
            parameters[:, :, 3] ** 2
            / (self.m * self.sigma**3 * self.epsilon_k)
            * 1e-19
            * (JOULE / KELVIN / KB)
        )
        self.kappa_ab = parameters[:, :, 4]
        self.epsilon_k_ab = parameters[:, :, 5]
        self.na = parameters[:, :, 6]
        self.nb = parameters[:, :, 7]
        self.parameters = parameters.detach().cpu().numpy()
        self.kij = kij
        self.kij_np = None if kij is None else kij.detach().cpu().numpy()

    def helmholtz_energy_density(self, temperature, density):
        # temperature dependent segment diameter
        d = self.sigma * (1 - 0.12 * (-3 * self.epsilon_k / temperature[:, None]).exp())

        zeta0 = np.pi / 6 * (self.m * density).sum(dim=1, keepdim=True)
        zeta1 = np.pi / 6 * (self.m * density * d).sum(dim=1, keepdim=True)
        zeta2 = np.pi / 6 * (self.m * density * d * d).sum(dim=1, keepdim=True)
        zeta3 = np.pi / 6 * (self.m * density * d * d * d).sum(dim=1, keepdim=True)

        zeta23 = zeta2 / zeta3
        zeta3_2 = zeta3 * zeta3
        zeta3_3 = zeta3_2 * zeta3
        zeta3_m1 = 1 / (1 - zeta3)
        zeta3_m2 = zeta3_m1 * zeta3_m1
        etas = [
            1,
            zeta3,
            zeta3_2,
            zeta3_3,
            zeta3_2 * zeta3_2,
            zeta3_2 * zeta3_3,
            zeta3_3 * zeta3_3,
        ]

        # hard sphere
        hs = (6 / np.pi) * (
            zeta1 * zeta2 * zeta3_m1 * 3
            + zeta2 * zeta2 * zeta3_m2 * zeta23
            + (zeta2 * zeta23 * zeta23 - zeta0) * (1 - zeta3).log()
        )

        # hard chain
        c = zeta2 * zeta3_m2
        g = zeta3_m1 + d * c * 1.5 - d * d * c * c * (zeta3 - 1) * 0.5
        hc = (-density * (self.m - 1) * g.log()).sum(dim=1, keepdim=True)

        # dispersion
        # mean segment number
        x = density / density.sum(dim=1, keepdim=True)
        m = (x * self.m).sum(dim=1, keepdim=True)

        # mixture densities, crosswise interactions of all segments on all chains
        n = self.m.shape[1]
        if self.kij is not None and n != 2:
            raise Exception("kij can only be used for binary mixtures!")
        rho1mix = 0
        rho2mix = 0
        for i in range(n):
            for j in range(n):
                eps_ij = (
                    self.epsilon_k[:, i] * self.epsilon_k[:, j]
                ).sqrt() / temperature
                if self.kij is not None and i != j:
                    eps_ij *= 1 - self.kij[:, 0]
                sigma_ij = (0.5 * (self.sigma[:, i] + self.sigma[:, j])) ** 3
                m_ij = self.m[:, i] * self.m[:, j]
                rhoij = density[:, i : i + 1] * density[:, j : j + 1]
                rhoij *= (m_ij * eps_ij * sigma_ij)[:, None]
                rho1mix += rhoij
                rho2mix += rhoij * eps_ij[:, None]

        I1 = 0
        I2 = 0
        m1 = (m - 1) / m
        m2 = m1 * (m - 2) / m
        for i in range(7):
            I1 += (m2 * A2[i] + m1 * A1[i] + A0[i]) * etas[i]
            I2 += (m2 * B2[i] + m1 * B1[i] + B0[i]) * etas[i]
        C1 = 1 / (
            1
            + m * (8 * zeta3 - 2 * zeta3_2) * zeta3_m2 * zeta3_m2
            + (1 - m)
            * (20 * zeta3 - 27 * zeta3_2 + 12 * zeta3_2 * zeta3 - 2 * zeta3_2 * zeta3_2)
            / ((1 - zeta3) * (1 - zeta3) * (2 - zeta3) * (2 - zeta3))
        )
        disp = (-rho1mix * 2 * I1 - rho2mix * C1 * I2 * m) * np.pi

        phi = hs + hc + disp

        # dipoles
        dipolar = torch.any(self.mu2 > 0, dim=1)
        etas = [1] + [e[dipolar, :] for e in etas[1:]]
        phi[dipolar, :] += self.phi_dipole(
            dipolar, temperature[dipolar], density[dipolar, :], etas
        )

        # association
        associating_comps = torch.count_nonzero(self.na + self.nb, dim=1)
        self_associating_comps = torch.count_nonzero(self.na * self.nb, dim=1)
        if torch.any(associating_comps > 2):
            raise Exception("Only up to two associating components are allowed!")

        self_associating = (associating_comps == 1) & (self_associating_comps == 1)
        phi[self_associating, :] += self.phi_self_assoc(
            self_associating,
            temperature[self_associating],
            density[self_associating, :],
            d[self_associating],
            zeta2[self_associating, :],
            zeta3_m1[self_associating, :],
        )

        cross_associating = (associating_comps == 2) & (self_associating_comps == 2)
        phi[cross_associating, :] += self.phi_cross_assoc(
            cross_associating,
            temperature[cross_associating],
            density[cross_associating, :],
            d[cross_associating],
            zeta2[cross_associating, :],
            zeta3_m1[cross_associating, :],
            self.kij[cross_associating, 1],
        )

        induced_associating = (associating_comps == 2) & (self_associating_comps == 1)
        phi[induced_associating, :] += self.phi_induced_assoc(
            induced_associating,
            temperature[induced_associating],
            density[induced_associating, :],
            d[induced_associating],
            zeta2[induced_associating, :],
            zeta3_m1[induced_associating, :],
        )

        return phi

    def phi_dipole(self, dipolar, temperature, density, etas):
        m = self.m[dipolar, :]
        sigma = self.sigma[dipolar, :]
        epsilon_k = self.epsilon_k[dipolar, :]
        mu2 = self.mu2[dipolar, :]

        n = m.shape[1]
        mu2_term = sigma**3 * epsilon_k * mu2 / temperature[:, None]
        phi2 = 0
        phi3 = 0
        for i in range(n):
            for j in range(i, n):
                sigma_ij_3 = (0.5 * (sigma[:, i] + sigma[:, j])) ** 3
                mij = (m[:, i].clamp(max=2) * m[:, j].clamp(max=2)).sqrt()
                mij1 = (mij - 1) / mij
                mij2 = mij1 * (mij - 2) / mij
                eps_ij_t = (epsilon_k[:, i] * epsilon_k[:, j]).sqrt() / temperature
                c = 1 if i == j else 2
                phi2 -= (
                    density[:, i : i + 1]
                    * density[:, j : j + 1]
                    * mu2_term[:, i : i + 1]
                    * mu2_term[:, j : j + 1]
                    * pair_integral(mij1, mij2, etas, eps_ij_t)
                    / sigma_ij_3[:, None]
                    * c
                )
                for k in range(j, n):
                    sigma_ij = 0.5 * (sigma[:, i] + sigma[:, j])
                    sigma_ik = 0.5 * (sigma[:, i] + sigma[:, k])
                    sigma_jk = 0.5 * (sigma[:, j] + sigma[:, k])
                    mijk = (
                        m[:, i].clamp(max=2)
                        * m[:, j].clamp(max=2)
                        * m[:, k].clamp(max=2)
                    ).pow(1 / 3)
                    mijk1 = (mijk - 1) / mijk
                    mijk2 = mijk1 * (mijk - 2) / mijk
                    c = {1: 1, 2: 3, 3: 6}[len({i, j, k})]
                    phi3 -= (
                        density[:, i : i + 1]
                        * density[:, j : j + 1]
                        * density[:, k : k + 1]
                        * mu2_term[:, i : i + 1]
                        * mu2_term[:, j : j + 1]
                        * mu2_term[:, k : k + 1]
                        * triplet_integral(mijk1, mijk2, etas)
                        / (sigma_ij * sigma_ik * sigma_jk)[:, None]
                        * c
                    )
        phi2 *= np.pi
        phi3 *= 4 / 3 * np.pi * np.pi
        return phi2 * phi2 / (phi2 - phi3)

    def phi_self_assoc(self, associating, temperature, density, d, zeta2, zeta3_m1):
        kappa_ab = self.kappa_ab[associating, :].sum(dim=1, keepdim=True)
        epsilon_k_ab = self.epsilon_k_ab[associating, :].sum(dim=1, keepdim=True)
        na = self.na[associating, :]
        nb = self.nb[associating, :]
        sigma = (na * self.sigma[associating, :]).sum(dim=1, keepdim=True) / na.sum(
            dim=1, keepdim=True
        )
        d = (na * d).sum(dim=1, keepdim=True) / na.sum(dim=1, keepdim=True)

        delta = association_strength(
            0,
            0,
            temperature,
            sigma,
            kappa_ab,
            epsilon_k_ab,
            None,
            d,
            zeta2,
            zeta3_m1,
        )
        rhoa = (na * density).sum(dim=1, keepdim=True)
        rhob = (nb * density).sum(dim=1, keepdim=True)

        aux = 1 + (rhoa - rhob) * delta
        sqrt = (aux * aux + 4 * rhob * delta).sqrt()
        xa = 2 / (sqrt + 1 + (rhob - rhoa) * delta)
        xb = 2 / (sqrt + 1 + (rhoa - rhob) * delta)
        return rhoa * (xa.log() - 0.5 * xa + 0.5) + rhob * (xb.log() - 0.5 * xb + 0.5)

    def phi_cross_assoc(
        self, associating, temperature, density, d, zeta2, zeta3_m1, epsilon_k_aibj
    ):
        sigma = self.sigma[associating, :]
        kappa_ab = self.kappa_ab[associating, :]
        epsilon_k_ab = self.epsilon_k_ab[associating, :]
        rhoa = density * self.na[associating, :]
        rhob = density * self.nb[associating, :]

        if sigma.shape[1] > 2:
            raise Exception("Cross associaion is only implemented for binary mixtures!")

        delta = lambda i, j: association_strength(
            i,
            j,
            temperature,
            sigma,
            kappa_ab,
            epsilon_k_ab,
            epsilon_k_aibj,
            d,
            zeta2,
            zeta3_m1,
        )
        d00 = delta(0, 0)
        d01 = delta(0, 1)
        d10 = delta(1, 0)
        d11 = delta(1, 1)

        xa0, xa1 = 0.2 * d00 / d00, 0.2 * d00 / d00
        for _ in range(50):
            xa0 = Dual2(xa0, 1, 0)
            xa1 = Dual2(xa1, 0, 1)

            xb0_i = 1 + xa0 * rhoa[:, 0:1] * d00 + xa1 * rhoa[:, 1:2] * d01
            xb1_i = 1 + xa0 * rhoa[:, 0:1] * d10 + xa1 * rhoa[:, 1:2] * d11

            f0 = (
                xa0
                - 1
                + xa0 / xb0_i * rhob[:, 0:1] * d00
                + xa0 / xb1_i * rhob[:, 1:2] * d01
            )
            f1 = (
                (xa1 - 1)
                + xa1 / xb0_i * rhob[:, 0:1] * d10
                + xa1 / xb1_i * rhob[:, 1:2] * d11
            )

            g0 = f0.re
            g1 = f1.re
            j00 = f0.eps1
            j01 = f0.eps2
            j10 = f1.eps1
            j11 = f1.eps2
            det = j00 * j11 - j01 * j10

            delta_xa0 = (j11 * g0 - j01 * g1) / det
            delta_xa1 = (-j10 * g0 + j00 * g1) / det
            xa0_old = xa0.re
            xa1_old = xa1.re
            xa0 = xa0_old - delta_xa0
            xa1 = xa1_old - delta_xa1
            for i in range(len(xa0)):
                if xa0[(i,)] < 0:
                    xa0[(i,)] = 0.2 * xa0_old[(i,)]
                if xa1[(i,)] < 0:
                    xa1[(i,)] = 0.2 * xa1_old[(i,)]

            if g0.norm() < 1e-10 and g1.norm() < 1e-10:
                break

        xb0 = 1 / (1 + xa0 * rhoa[:, 0:1] * d00 + xa1 * rhoa[:, 1:2] * d01)
        xb1 = 1 / (1 + xa0 * rhoa[:, 0:1] * d10 + xa1 * rhoa[:, 1:2] * d11)
        f = lambda x: x.log() - 0.5 * x + 0.5
        return (
            rhoa[:, 0:1] * f(xa0)
            + rhoa[:, 1:2] * f(xa1)
            + rhob[:, 0:1] * f(xb0)
            + rhob[:, 1:2] * f(xb1)
        )

    # WARNING! Hardcoded für nA=0
    def phi_induced_assoc(self, associating, temperature, density, d, zeta2, zeta3_m1):
        sigma = self.sigma[associating, :]
        kappa_ab = self.kappa_ab[associating, :]
        epsilon_k_ab = self.epsilon_k_ab[associating, :]
        na0 = self.na[associating, 0:1]
        na1 = self.na[associating, 1:2]
        nb0 = self.nb[associating, 0:1]
        nb1 = self.nb[associating, 1:2]

        is_assoc = (kappa_ab * epsilon_k_ab).sign()
        n = is_assoc.shape[1]

        if n > 2:
            raise Exception(
                "Induced association is only implemented for binary mixtures!"
            )

        delta_rho = (
            lambda i, j: association_strength(
                i,
                j,
                temperature,
                sigma,
                kappa_ab,
                epsilon_k_ab,
                None,
                d,
                zeta2,
                zeta3_m1,
            )
            * density[:, j : j + 1]
        )
        d00 = delta_rho(0, 0)
        d01 = delta_rho(0, 1)
        d10 = delta_rho(1, 0)
        d11 = delta_rho(1, 1)

        xa = 0.2 * d00 / d00

        for _ in range(50):
            xa = Dual2(xa, 1, 0)
            xb0_i = 1 + xa * (na0 * d00 + na1 * d01)
            xb1_i = 1 + xa * (na0 * d10 + na1 * d11)
            f0 = (
                xa * (xb0_i * xb1_i + nb0 * xb1_i * d00 + nb1 * xb0_i * d01)
                - xb0_i * xb1_i
            )
            f1 = (
                xa * (xb0_i * xb1_i + nb0 * xb1_i * d10 + nb1 * xb0_i * d11)
                - xb0_i * xb1_i
            )
            f = na0 * f0 + na1 * f1

            delta_x = f.re / f.eps1
            xa_old = xa.re
            xa = xa_old - delta_x
            for i in range(len(xa)):
                if xa[(i,)] < 0:
                    xa[(i,)] = 0.2 * xa_old[(i,)]

            if f.re.norm() < 1e-10:
                break

        xb0 = 1 / (1 + xa * (na0 * d00 + na1 * d01))
        xb1 = 1 / (1 + xa * (na0 * d10 + na1 * d11))

        f = lambda x: x.log() - 0.5 * x + 0.5
        return density[:, 0:1] * (f(xa) * na0 + f(xb0) * nb0) + density[:, 1:2] * (
            f(xa) * na1 + f(xb1) * nb1
        )

    def derivatives(self, temperature, density):
        (k, n) = density.shape
        volume = DualTensor(
            torch.ones_like(temperature)[:, None],
            torch.zeros((k, 1, n + 1), device=temperature.device, dtype=torch.float64),
            torch.ones_like(temperature)[:, None],
            torch.zeros((k, 1, n + 1), device=temperature.device, dtype=torch.float64),
        )
        volume.eps1[:, :, n] = 1
        moles = DualTensor(
            density,
            torch.zeros((k, n, n + 1), device=temperature.device, dtype=torch.float64),
            torch.zeros((k, n), device=temperature.device, dtype=torch.float64),
            torch.zeros((k, n, n + 1), device=temperature.device, dtype=torch.float64),
        )
        for i in range(n):
            moles.eps1[:, i, i] = 1

        a = self.helmholtz_energy_density(temperature, moles / volume) * volume
        p = density.sum(dim=1) - a.eps2[:, 0]
        mu = a.eps1[:, 0, :n]
        v = -(1 - a.eps1eps2[:, 0, :n]) / (
            -density.sum(dim=1, keepdim=True) - a.eps1eps2[:, 0, n : n + 1]
        )
        a = a.re[:, 0]
        return a, p, mu, v

    def bubble_point(self, temperature, liquid_molefracs, pressure):
        density, nans = PcSaft.bubble_point(
            self.parameters,
            self.kij_np,
            temperature.detach().cpu().numpy(),
            liquid_molefracs.detach().cpu().numpy(),
            pressure.detach().cpu().numpy(),
        )
        density = torch.from_numpy(density).to(self.m.device)
        nans = torch.from_numpy(nans).to(self.m.device)
        temperature = temperature[~nans]
        self.reduce(nans)

        rho_i_V = density[:, 0:2]
        rho_i_L = density[:, 2:4]
        rho_V = rho_i_V.sum(dim=1)
        y = rho_i_V / rho_i_V.sum(dim=1, keepdim=True)
        _, p_L, mu_i_L, v_i_L = self.derivatives(temperature, rho_i_L)
        a_V = self.helmholtz_energy_density(temperature, rho_i_V)[:, 0] / rho_V
        v_L = (y * v_i_L).sum(dim=1)
        g_L = (y * ((rho_i_V / rho_i_L).log() - mu_i_L)).sum(dim=1)
        p = -(a_V + p_L * v_L + g_L - 1) / (1 / rho_V - v_L)
        return p * temperature * (KB * KELVIN / ANGSTROM**3 / PASCAL), nans

    def dew_point(self, temperature, vapor_molefracs, pressure):
        density, nans = PcSaft.dew_point(
            self.parameters,
            self.kij_np,
            temperature.detach().cpu().numpy(),
            vapor_molefracs.detach().cpu().numpy(),
            pressure.detach().cpu().numpy(),
        )
        density = torch.from_numpy(density).to(self.m.device)
        nans = torch.from_numpy(nans).to(self.m.device)
        temperature = temperature[~nans]
        self.reduce(nans)

        rho_i_V = density[:, 0:2]
        rho_i_L = density[:, 2:4]
        rho_L = rho_i_L.sum(dim=1)
        x = rho_i_L / rho_i_L.sum(dim=1, keepdim=True)
        _, p_V, mu_i_V, v_i_V = self.derivatives(temperature, rho_i_V)
        a_L = self.helmholtz_energy_density(temperature, rho_i_L)[:, 0] / rho_L
        v_V = (x * v_i_V).sum(dim=1)
        g_V = (x * ((rho_i_L / rho_i_V).log() - mu_i_V)).sum(dim=1)
        p = -(a_L + p_V * v_V + g_V - 1) / (1 / rho_L - v_V)
        return p * temperature * (KB * KELVIN / ANGSTROM**3 / PASCAL), nans

    def reduce(self, nans):
        self.m = self.m[~nans]
        self.sigma = self.sigma[~nans]
        self.epsilon_k = self.epsilon_k[~nans]
        self.mu2 = self.mu2[~nans]
        self.kappa_ab = self.kappa_ab[~nans]
        self.epsilon_k_ab = self.epsilon_k_ab[~nans]
        self.kij = self.kij[~nans]
        self.na = self.na[~nans]
        self.nb = self.nb[~nans]


def pair_integral(mij1, mij2, etas, eps_ij_t):
    return sum(
        eta
        * (
            (eps_ij_t * (bi[0] + mij1 * bi[1] + mij2 * bi[2]))
            + (ai[0] + mij1 * ai[1] + mij2 * ai[2])
        )[:, None]
        for eta, ai, bi in zip(etas, AD, BD)
    )


def triplet_integral(mijk1, mijk2, etas):
    return sum(
        eta * (ci[0] + mijk1 * ci[1] + mijk2 * ci[2])[:, None]
        for eta, ci in zip(etas, CD)
    )


def association_strength(
    i, j, temperature, sigma, kappa_ab, epsilon_k_ab, epsilon_k_aibj, d, zeta2, zeta3_m1
):
    di = d[:, i : i + 1]
    dj = d[:, j : j + 1]
    k = di * dj / (di + dj) * zeta2 * zeta3_m1
    sigma3_kappa_aibj = (sigma[:, i] * sigma[:, j]) ** 1.5 * (
        kappa_ab[:, i] * kappa_ab[:, j]
    ).sqrt()
    if epsilon_k_aibj is not None and i != j:
        _epsilon_k_aibj = torch.where(
            epsilon_k_aibj != 0,
            epsilon_k_aibj,
            0.5 * (epsilon_k_ab[:, i] + epsilon_k_ab[:, j]),
        )
    else:
        _epsilon_k_aibj = 0.5 * (epsilon_k_ab[:, i] + epsilon_k_ab[:, j])
    return (
        zeta3_m1
        * (k * (2 * k + 3) + 1)
        * sigma3_kappa_aibj[:, None]
        * ((_epsilon_k_aibj[:, None] / temperature[:, None]).exp() - 1)
    )
