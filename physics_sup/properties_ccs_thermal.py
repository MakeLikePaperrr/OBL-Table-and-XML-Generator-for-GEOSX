import numpy as np
from darts.physics import *
# from select_para import *
# from EOS.fugacity_activity import *
# from EOS.k_update import *
# from properties_basic import *

from flash import NPhaseSSI
from flash import AQProperties, VLProperties, HProperties
from flash import VdWP


class ssiFlash:
    def __init__(self, components, eos, phases):
        self.components = components
        self.nc = len(components)
        self.phases = phases
        self.nph = len(phases)
        self.eos = eos
        self.F = NPhaseSSI(self.components, self.eos, self.phases)

    def evaluate(self, pressure, temperature, zc):
        phases = self.F.runFlash(pressure, temperature, zc)

        x = np.zeros((self.nph, self.nc))
        V = np.zeros(self.nph)

        k = 0
        for j in range(self.nph):
            if self.phases[j] == phases[k]:
                V[j] = self.F.V[k]
                for i in range(self.nc):
                    x[j, i] = self.F.x[k*self.nc + i]
                k += 1
                if k > len(phases)-1:
                    break

        return V, x


class ssiFlashKinHydrate(ssiFlash):
    def __init__(self, components, eos, phases, nhp=0, nhc=0):
        # Number of hydrate phases and hydrate components
        self.comp_norm = components[0:len(components)-nhc]
        self.ph_norm = phases[0:len(phases)-nhp]
        self.ph_H = phases[(len(phases)-nhp):]
        self.nhp = nhp
        self.nhc = nhc

        super().__init__(self.comp_norm, eos, self.ph_norm)

    def renormalize_H(self, v, x, zh):
        # Include hydrate components again
        V = np.zeros(self.nph + self.nhp)
        X = np.zeros((self.nph + self.nhp, self.nc + self.nhc))

        for j in range(self.nph):
            V[j] = v[j] * (1 - np.sum(zh))
            for i in range(self.nc):
                X[j, i] = x[j, i]

        for j in range(self.nhp):
            V[self.nph + j] = np.sum(zh)
            if np.sum(zh) > 0:
                X[self.nph + j, self.nc:] = zh/np.sum(zh)

        return V, X

    def evaluate(self, pressure, temperature, zc):
        # Exclude hydrate components
        zh = zc[self.nc:]
        zc_norm = zc[:self.nc] / (1 - np.sum(zh))

        # Run regular flash
        v, x = super().evaluate(pressure, temperature, zc_norm)

        # Include hydrate components again
        V, x = self.renormalize_H(v, x, zh)

        return V, x


# region Aq
class AqProperties:
    def __init__(self, species, eos="AQ3"):
        self.prop_container = AQProperties(species, eos)


class AqDensity(AqProperties):
    def __init__(self, species):
        super().__init__(species)

    def evaluate(self, p, T, x):
        return self.prop_container.density(p, T, x)


class AqViscosity(AqProperties):
    def __init__(self, species):
        super().__init__(species)

    def evaluate(self, p, T, x, rho):
        return self.prop_container.viscosity(p, T, x, rho)


class AqEnthalpy(AqProperties):
    def __init__(self, species):
        super().__init__(species)

    def evaluate(self, p, T, x):
        return self.prop_container.enthalpy(p, T, x)


class AqConductivity(AqProperties):
    def __init__(self, species):
        super().__init__(species)

    def evaluate(self, p, T, x, rho):
        return self.prop_container.conductivity(p, T, x, rho)
# endregion


# region VL
class VlProperties:
    def __init__(self, species, eos):
        self.prop_container = VLProperties(species, eos)


class VLDensity(VlProperties):
    def __init__(self, species, eos):
        super().__init__(species, eos)

    def evaluate(self, p, T, x):
        return self.prop_container.density(p, T, x)


class VViscosity(VlProperties):
    def __init__(self, species, eos):
        super().__init__(species, eos)

    def evaluate(self, p, T, x, rho):
        return self.prop_container.viscosity(p, T, x, rho)


class VLEnthalpy(VlProperties):
    def __init__(self, species, eos):
        super().__init__(species, eos)

    def evaluate(self, p, T, x):
        return self.prop_container.enthalpy(p, T, x)


class VConductivity(VlProperties):
    def __init__(self, species, eos):
        super().__init__(species, eos)

    def evaluate(self, p, T, x, rho):
        return self.prop_container.conductivity(p, T, x, rho)
# endregion


# region H
class HydrateProperties:
    def __init__(self, flash_container: ssiFlashKinHydrate, hydrate_type: str):
        self.flash = flash_container
        comp_norm = flash_container.comp_norm
        eos = flash_container.eos[hydrate_type]

        self.prop = HProperties(comp_norm, eos, hydrate_type)
        self.H_eos = VdWP(comp_norm, hydrate_type)


class HydrateDensity(HydrateProperties):
    def __init__(self, flash_container, hydrate_type):
        super().__init__(flash_container, hydrate_type)

    def evaluate(self, p, T, x):
        f0 = self.flash.F.f
        return self.prop.density(p, T, f0)


class HydrateEnthalpy(HydrateProperties):
    def __init__(self, flash_container, hydrate_type):
        super().__init__(flash_container, hydrate_type)

    def evaluate(self, p, T, x):
        f0 = self.flash.F.f
        return self.prop.enthalpy(p, T, f0)


class HydrateConductivity(HydrateProperties):
    def __init__(self, flash_container, hydrate_type):
        super().__init__(flash_container, hydrate_type)

    def evaluate(self, p, T, x, rho):
        f0 = self.flash.F.f
        return self.prop.conductivity(p, T, f0, rho)


class HydrateReactionRate(HydrateProperties):
    def __init__(self, flash_container, hydrate_type, perm, poro, stoich=[]):
        super().__init__(flash_container, hydrate_type)

        # input parameters for kinetic rate
        self.stoich = stoich
        self.perm = perm
        self.poro = poro

        self.water_idx = self.flash.components.index("H2O")

    def calc_df(self, pressure, temperature):
        # Calculate fugacity difference
        f0 = self.flash.F.f

        self.H_eos.parameters(pressure, temperature)
        fwH = self.H_eos.fwH(f0)

        df = fwH - f0[self.water_idx]  # if df < 0 formation, if df > 0 dissociation
        xH = self.H_eos.xH()

        return df, xH

    def evaluate(self, pressure, temperature, sh):
        # Calculate fugacity difference and hydrate composition
        df, xH = self.calc_df(pressure, temperature)

        # Reaction rate following Yin (2018)
        # Constants needs to be determined through history matching with experiments
        K = 3.11E14  # reaction constant [mol/(m^2 bar day)]

        # surface area following Yin (2018)
        F_A = 1
        perm = self.perm * 1E-15  # mD to m2
        r_p = (45 * perm * (1 - self.poro) ** 2 / (self.poro ** 3)) ** (1 / 2)
        A_s = 0.879 * F_A * (1 - self.poro) / r_p * sh ** (2 / 3)  # hydrate surface area [m2]

        # Thermodynamic parameters
        dE = -81E3  # activation energy [J/mol]
        R = 8.3145  # gas constant [J/(K.mol)]

        # # K is reaction cons, A_s hydrate surface area, dE activation energy, driving force is fugacity difference
        kinetic_rate = K * A_s * np.exp(dE / (R * temperature)) * df

        return kinetic_rate, self.stoich


# class HydrateReactionEnthalpy:
#     def __init__(self):
#         nH = 5.75
#         Mw = [18.015, 16.043]
#         self.mH = nH * Mw[0] + Mw[1]  # g/mol -> kg/kmol
#
#     def evaluate(self, temperature):
#         Cf = 33.72995  # J/kg cal/gmol
#         if temperature-273.15 > 0:
#             (C1, C2) = (13521, -4.02)
#         else:
#             (C1, C2) = (6534, -11.97)
#         en = Cf * (C1 + C2 / temperature)/1000  # kJ/kg
#
#         return en * self.mH  # kJ/kg * kg/kmol -> kJ/kmol
# endregion


class GasViscosity(property_evaluator_iface):
    def __init__(self):
        super().__init__()
        self.A_CO2 = [-1.146067e-01, 6.978380e-07, 3.976765e-10, 6.336120e-02,
                      -1.166119e-02, 7.142596e-04, 6.519333e-06, -3.567559e-01, 3.180473e-02]
        self.A_C1 = [-2.25711259e-02, -1.31338399e-04, 3.44353097e-06, -4.69476607e-08, 2.23030860e-02,
                     -5.56421194e-03, 2.90880717e-05, -1.90511457e0, 1.14082882e0, -2.25890087e-01]
        self.pc_c1 = props('C1', 'Pc')
        self.Tc_c1 = props('C1', 'Tc')
        self.Mw_C1 = props('C1', 'Mw')
        self.Mw_CO2 = props('CO2', 'Mw')

    def evaluate(self, state, y):  # Ignores CH4 concentration
        T = state[3]
        p = state[0]
        pr = p / self.pc_c1
        Tr = T / self.Tc_c1

        mu_co2 = (self.A_CO2[0] + self.A_CO2[1] * p + self.A_CO2[2] * p ** 2 + self.A_CO2[3] * np.log(T) + self.A_CO2[
            4] * np.log(T) ** 2 + self.A_CO2[5] * np.log(
            T) ** 3) / (1 + self.A_CO2[6] * p + self.A_CO2[7] * np.log(T) + self.A_CO2[8] * np.log(T) ** 2)

        if 72 <= p <= 77:
            mu_co2 = (p - 72) / 5 * mu_co2 + (77 - p) / 5 * 0.018
        elif p < 73:
            mu_co2 = 0.018

        mu_c1 = (self.A_C1[0] + self.A_C1[1] * pr + self.A_C1[2] * pr ** 2 + self.A_C1[3] * pr ** 3 + self.A_C1[
            4] * Tr + self.A_C1[5] * Tr ** 2) / (
                        1 + self.A_C1[6] * pr + self.A_C1[7] * Tr + self.A_C1[8] * Tr ** 2 + self.A_C1[9] * Tr ** 3)

        mu_g = (mu_c1 * y[1] * np.sqrt(self.Mw_C1) + mu_co2 * y[0] * np.sqrt(self.Mw_CO2)) / (
        (y[1] * np.sqrt(self.Mw_C1) + y[0] * np.sqrt(self.Mw_CO2)))

        return mu_g


class RockCompactionEvaluator(property_evaluator_iface):
    def __init__(self, pref=1, compres=1.45e-5):
        super().__init__()
        self.Pref = pref
        self.compres = compres

    def evaluate(self, state):
        pressure = state[0]

        return (1.0 + self.compres * (pressure - self.Pref))


class RockEnergyEvaluator(property_evaluator_iface):
    def __init__(self):
        super().__init__()

    def evaluate(self, temperature):
        T_ref = 273.15
        # c_vr = 3710  # 1400 J/kg.K * 2650 kg/m3 -> kJ/m3
        c_vr = 1  # 1400 J/kg.K * 2650 kg/m3 -> kJ/m3

        return c_vr * (temperature - T_ref)  # kJ/m3
