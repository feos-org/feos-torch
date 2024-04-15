import torch
import numpy as np
from collections import Counter

from si_units import KELVIN, KB, ANGSTROM, PASCAL, JOULE
from feos_torch import GcPcSaft


from .dual_torch import Dual2, DualTensor
from .pcsaft_pure import A0, A1, A2, B0, B1, B2, AD, BD, CD


class GcPcSaftMix:
    def __init__(
        self,
        segment_identifier,
        parameter,
        segment_lists,
        bond_lists,
        binary_segment_records,
        phi=None,
    ):
        m, sigma, epsilon_k, mu, kappa_ab, epsilon_k_ab, na, nb = parameter
        segment_indices = dict((s, i) for i, s in enumerate(segment_identifier))

        segments = [[Counter(s) for s in seg] for seg in segment_lists]
        segments = [
            [[s[i] for i in segment_identifier] for s in seg] for seg in segments
        ]
        segments = torch.tensor(segments, dtype=torch.float64)

        bonds = [
            [
                Counter(
                    tuple(sorted((segment_indices[s[i]], segment_indices[s[j]]))[::-1])
                    for i, j in b
                )
                for s, b in zip(seg, bon)
            ]
            for seg, bon in zip(segment_lists, bond_lists)
        ]
        bonds = [
            [
                [
                    [b[(i, j)] for j in range(len(segment_identifier))]
                    for i in range(len(segment_identifier))
                ]
                for b in bon
            ]
            for bon in bonds
        ]
        self.bonds = torch.tensor(bonds, dtype=torch.float64)

        # basic parameters
        self.m = segments * m
        self.sigma = sigma
        self.epsilon_k = epsilon_k
        self.phi = phi

        self.kab = torch.zeros([len(segment_identifier)] * 2, dtype=torch.float64)
        for s1, s2, k in binary_segment_records:
            self.kab[segment_indices[s1], segment_indices[s2]] = k
            self.kab[segment_indices[s2], segment_indices[s1]] = k

        # dipole parameters
        self.m_mix = (segments * m).sum(dim=2)
        self.sigma_mix = ((segments * m * self.sigma**3).sum(dim=2) / self.m_mix) ** (
            1 / 3
        )
        self.epsilon_k_mix = (segments * m * self.epsilon_k).sum(dim=2) / self.m_mix
        self.mu2 = (
            (segments * mu**2).sum(dim=2) / self.m_mix * (1e-19 * JOULE / KELVIN / KB)
        )

        # association parameters
        self.is_assoc = segments * (kappa_ab * epsilon_k_ab).sign()
        if torch.any(self.is_assoc.sum(dim=2) > 1):
            raise Exception(
                "Only up to one associating segment per component is allowed!"
            )
        self.sigma_assoc = (self.is_assoc * self.sigma).sum(dim=2)
        self.epsilon_k_assoc = (self.is_assoc * self.epsilon_k).sum(dim=2)
        self.kappa_ab = (segments * kappa_ab).sum(dim=2)
        self.epsilon_k_ab = (segments * epsilon_k_ab).sum(dim=2)
        self.na = (segments * na).sum(dim=2)
        self.nb = (segments * nb).sum(dim=2)

        # FeOs interface
        phi_np = np.ones_like(self.m_mix) if phi is None else phi.detach().numpy()
        segment_records = [
            (
                ident,
                np.array(
                    [
                        m[i].item(),
                        sigma[i].item(),
                        epsilon_k[i].item(),
                        mu[i].item(),
                        kappa_ab[i].item(),
                        epsilon_k_ab[i].item(),
                        na[i].item(),
                        nb[i].item(),
                    ]
                ),
            )
            for i, ident in enumerate(segment_identifier)
        ]
        self.gc_pcsaft = GcPcSaft(
            segment_records,
            segment_lists,
            bond_lists,
            binary_segment_records,
            phi_np,
        )

    def helmholtz_energy_density(self, temperature, density, kab=None):
        # temperature dependent segment diameter
        d = self.sigma * (
            1 - 0.12 * (-3 * self.epsilon_k / temperature[:, None, None]).exp()
        )

        zeta0 = np.pi / 6 * (self.m.sum(dim=2) * density).sum(dim=1, keepdim=True)
        zeta1 = np.pi / 6 * ((self.m * d).sum(dim=2) * density).sum(dim=1, keepdim=True)
        zeta2 = (
            np.pi / 6 * ((self.m * d * d).sum(dim=2) * density).sum(dim=1, keepdim=True)
        )
        zeta3 = (
            np.pi
            / 6
            * ((self.m * d * d * d).sum(dim=2) * density).sum(dim=1, keepdim=True)
        )

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
        hc = 0
        for alpha in range(len(self.sigma)):
            for beta in range(alpha + 1):
                da, db = d[:, :, alpha], d[:, :, beta]
                cdab = c * ((da * db) / (da + db))
                g = zeta3_m1 + cdab * 3 - cdab * cdab * (zeta3 - 1) * 2
                hc -= (density * self.bonds[:, :, alpha, beta] * g.log()).sum(
                    dim=1, keepdim=True
                )

        # dispersion
        # mean segment number
        x = density / density.sum(dim=1, keepdim=True)
        m = (x * self.m.sum(dim=2)).sum(dim=1, keepdim=True)

        # mixture densities, crosswise interactions of all segments on all chains
        n = self.m.shape[1]
        s = self.m.shape[2]
        rho1mix = 0
        rho2mix = 0
        for i in range(n):
            for j in range(n):
                for alpha in range(s):
                    for beta in range(s):
                        eps_ab = self.epsilon_k[alpha] * self.epsilon_k[beta]
                        if self.phi is not None:
                            eps_ab = (
                                eps_ab * self.phi[:, i : i + 1] * self.phi[:, j : j + 1]
                            )
                        eps_ab = eps_ab.sqrt() / temperature[:, None]
                        if i != j:
                            eps_ab *= 1 - self.kab[alpha, beta]
                        sigma_ab = (0.5 * (self.sigma[alpha] + self.sigma[beta])) ** 3
                        m_ab = self.m[:, i : i + 1, alpha] * self.m[:, j : j + 1, beta]
                        rhoij = density[:, i : i + 1] * density[:, j : j + 1]
                        rhoij *= m_ab * eps_ab * sigma_ab
                        rho1mix += rhoij
                        rho2mix += rhoij * eps_ab

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
        associating_segm = torch.count_nonzero(self.kappa_ab * self.epsilon_k_ab, dim=1)
        self_associating_segm = torch.count_nonzero(self.na * self.nb, dim=1)
        if torch.any(associating_segm > 2):
            raise Exception("Only up to two associating segments are allowed!")

        self_associating = (associating_segm == 1) & (self_associating_segm == 1)
        phi[self_associating, :] += self.phi_assoc(
            self_associating,
            temperature[self_associating],
            density[self_associating, :],
            zeta2[self_associating, :],
            zeta3_m1[self_associating, :],
        )

        cross_associating = (associating_segm == 2) & (self_associating_segm == 2)
        phi[cross_associating, :] += self.phi_cross_assoc(
            cross_associating,
            temperature[cross_associating],
            density[cross_associating, :],
            zeta2[cross_associating, :],
            zeta3_m1[cross_associating, :],
        )

        induced_associating = (associating_segm == 2) & (self_associating_segm == 1)
        phi[induced_associating, :] += self.phi_induced_assoc(
            induced_associating,
            temperature[induced_associating],
            density[induced_associating, :],
            zeta2[induced_associating, :],
            zeta3_m1[induced_associating, :],
        )

        return phi

    def phi_dipole(self, dipolar, temperature, density, etas):
        m = self.m_mix[dipolar, :]
        sigma = self.sigma_mix[dipolar, :]
        epsilon_k = self.epsilon_k_mix[dipolar, :]
        mu2 = self.mu2[dipolar, :]

        n = m.shape[1]
        mu2_term = mu2 / temperature[:, None]
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

    def phi_assoc(self, associating, temperature, density, zeta2, zeta3_m1):
        sigma = self.sigma_assoc[associating, :].sum(dim=1, keepdim=True)
        epsilon_k = self.epsilon_k_assoc[associating, :].sum(dim=1, keepdim=True)
        kappa_ab = self.kappa_ab[associating, :].sum(dim=1, keepdim=True)
        epsilon_k_ab = self.epsilon_k_ab[associating, :].sum(dim=1, keepdim=True)

        d = sigma * (1 - 0.12 * (-3 * epsilon_k / temperature[:, None]).exp())

        is_assoc = (
            self.kappa_ab[associating, :] * self.epsilon_k_ab[associating, :]
        ).sign()
        rho = (is_assoc * density).sum(dim=1, keepdim=True)

        deltarho = (
            association_strength(
                0, 0, temperature, sigma, kappa_ab, epsilon_k_ab, d, zeta2, zeta3_m1
            )
            * rho
        )

        xa = 2 / ((1 + 4 * deltarho).sqrt() + 1)
        return rho * (2 * xa.log() - xa + 1)

    # WARNING! Hardcoded for nA=nB=1
    def phi_cross_assoc(self, associating, temperature, density, zeta2, zeta3_m1):
        sigma = self.sigma_assoc[associating, :]
        epsilon_k = self.epsilon_k_assoc[associating, :]
        kappa_ab = self.kappa_ab[associating, :]
        epsilon_k_ab = self.epsilon_k_ab[associating, :]

        d = sigma * (1 - 0.12 * (-3 * epsilon_k / temperature[:, None]).exp())

        is_assoc = (kappa_ab * epsilon_k_ab).sign()
        n = is_assoc.shape[1]

        if n > 2:
            raise Exception("Cross associaion is only implemented for binary mixtures!")

        delta_rho = (
            lambda i, j: association_strength(
                i, j, temperature, sigma, kappa_ab, epsilon_k_ab, d, zeta2, zeta3_m1
            )
            * density[:, j : j + 1]
        )
        d00 = delta_rho(0, 0)
        d01 = delta_rho(0, 1)
        d10 = delta_rho(1, 0)
        d11 = delta_rho(1, 1)

        xa0, xa1 = 0.2, 0.2

        for _ in range(50):
            xa0 = Dual2(xa0, 1, 0)
            xa1 = Dual2(xa1, 0, 1)
            f0 = xa0 + xa0 * xa0 * d00 + xa0 * xa1 * d01 - 1
            f1 = xa1 + xa1 * xa0 * d10 + xa1 * xa1 * d11 - 1

            g0 = f0.re
            g1 = f1.re
            j00 = f0.eps1
            j01 = f0.eps2
            j10 = f1.eps1
            j11 = f1.eps2
            det = j00 * j11 - j01 * j10
            xa0 = xa0.re - (j11 * g0 - j01 * g1) / det
            xa1 = xa1.re - (-j10 * g0 + j00 * g1) / det

            if g0.norm() < 1e-10 and g1.norm() < 1e-10:
                break

        f = lambda x: 2 * x.log() - x + 1
        return density[:, 0:1] * f(xa0) + density[:, 1:2] * f(xa1)

    # WARNING! Hardcoded für nA=0
    def phi_induced_assoc(self, associating, temperature, density, zeta2, zeta3_m1):
        sigma = self.sigma_assoc[associating, :]
        epsilon_k = self.epsilon_k_assoc[associating, :]
        kappa_ab = self.kappa_ab[associating, :]
        epsilon_k_ab = self.epsilon_k_ab[associating, :]
        na0 = self.na[associating, 0:1]
        na1 = self.na[associating, 1:2]
        nb0 = self.nb[associating, 0:1]
        nb1 = self.nb[associating, 1:2]

        d = sigma * (1 - 0.12 * (-3 * epsilon_k / temperature[:, None]).exp())

        is_assoc = (kappa_ab * epsilon_k_ab).sign()
        n = is_assoc.shape[1]

        if n > 2:
            raise Exception(
                "Induced association is only implemented for binary mixtures!"
            )

        delta_rho = (
            lambda i, j: association_strength(
                i, j, temperature, sigma, kappa_ab, epsilon_k_ab, d, zeta2, zeta3_m1
            )
            * density[:, j : j + 1]
        )
        d00 = delta_rho(0, 0)
        d01 = delta_rho(0, 1)
        d10 = delta_rho(1, 0)
        d11 = delta_rho(1, 1)

        xa = 0.2

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

            xa = xa.re - f.re / f.eps1

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
            torch.zeros((k, 1, n + 1), dtype=torch.float64),
            torch.ones_like(temperature)[:, None],
            torch.zeros((k, 1, n + 1), dtype=torch.float64),
        )
        volume.eps1[:, :, n] = 1
        moles = DualTensor(
            density,
            torch.zeros((k, n, n + 1), dtype=torch.float64),
            torch.zeros((k, n), dtype=torch.float64),
            torch.zeros((k, n, n + 1), dtype=torch.float64),
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
        density, nans = self.gc_pcsaft.bubble_point(
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
        g_L = (y * (np.log(rho_i_V / rho_i_L) - mu_i_L)).sum(dim=1)
        p = -(a_V + p_L * v_L + g_L - 1) / (1 / rho_V - v_L)
        return p * temperature * (KB * KELVIN / ANGSTROM**3 / PASCAL), nans

    def dew_point(self, temperature, vapor_molefracs, pressure):
        density, nans = self.gc_pcsaft.dew_point(
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
        g_V = (x * (np.log(rho_i_L / rho_i_V) - mu_i_V)).sum(dim=1)
        p = -(a_L + p_V * v_V + g_V - 1) / (1 / rho_L - v_V)
        return p * temperature * (KB * KELVIN / ANGSTROM**3 / PASCAL), nans

    def reduce(self, nans):
        self.m = self.m[~nans, :]
        self.m_mix = self.m_mix[~nans, :]
        self.sigma_mix = self.sigma_mix[~nans, :]
        self.epsilon_k_mix = self.epsilon_k_mix[~nans, :]
        self.mu2 = self.mu2[~nans]
        self.is_assoc = self.is_assoc[~nans]
        self.kappa_ab = self.kappa_ab[~nans]
        self.epsilon_k_ab = self.epsilon_k_ab[~nans]
        self.na = self.na[~nans]
        self.nb = self.nb[~nans]
        self.sigma_assoc = self.sigma_assoc[~nans, :]
        self.epsilon_k_assoc = self.epsilon_k_assoc[~nans, :]
        self.bonds = self.bonds[~nans]
        self.phi = self.phi[~nans, :]


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
    i, j, temperature, sigma, kappa_ab, epsilon_k_ab, d, zeta2, zeta3_m1
):
    di = d[:, i : i + 1]
    dj = d[:, j : j + 1]
    k = di * dj / (di + dj) * zeta2 * zeta3_m1
    sigma3_kappa_aibj = (sigma[:, i] * sigma[:, j]) ** 1.5 * (
        kappa_ab[:, i] * kappa_ab[:, j]
    ).sqrt()
    epsilon_k_aibj = 0.5 * (epsilon_k_ab[:, i] + epsilon_k_ab[:, j])
    return (
        zeta3_m1
        * (k * (2 * k + 3) + 1)
        * sigma3_kappa_aibj[:, None]
        * ((epsilon_k_aibj[:, None] / temperature[:, None]).exp() - 1)
    )
