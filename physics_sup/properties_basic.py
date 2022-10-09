import numpy as np

# from src.cubic_main import *
# from src.Binary_Interactions import *
# from src.flash_funcs import *


#  dummy function
class const_fun():
    def __init__(self, value=0):
        super().__init__()
        self.ret = value

    def evaluate(self, dummy1=0, dummy2=0, dummy3=0):
        return self.ret

class flash_3phase():
    def __init__(self, components, T):
        self.components = components
        self.T = T
        mixture = Mix(components)
        binary = Kij(components)
        mixture.kij_cubic(binary)

        self.eos = preos(mixture, mixrule='qmr', volume_translation=True)

    def evaluate(self, p, zc):
        nu, x, status = multiphase_flash(self.components, zc, self.T, p, self.eos)

        return x, nu


# Uncomment these two lines if numba package is installed and make things happen much faster:
from numba import jit
@jit(nopython=True)
def RR_func(zc, k, eps):
    a = 1 / (1 - np.max(k)) + eps
    b = 1 / (1 - np.min(k)) - eps

    max_iter = 200  # use enough iterations for V to converge
    for i in range(1, max_iter):
        V = 0.5 * (a + b)
        r = np.sum(zc * (k - 1) / (V * (k - 1) + 1))
        if abs(r) < 1e-12:
            break

        if r > 0:
            a = V
        else:
            b = V

    if i >= max_iter:
        print("Flash warning!!!")

    x = zc / (V * (k - 1) + 1)
    y = k * x

    return (x, y, V)

class Flash:
    def __init__(self, components, ki, min_z=1e-11):
        self.components = components
        self.min_z = min_z
        self.K_values = np.array(ki)

    def evaluate(self, pressure, zc):

        (x, y, V) = RR_func(zc, self.K_values, self.min_z)
        return np.array([y, x]), np.array([V, 1-V])


from flash import PR, SRK, AQ1, AQ3, VdWP
class EoS:
    def __init__(self, components, ions, eos):
        self.components = components
        self.NC = len(components)
        self.NI = len(ions)
        species = components + ions

        self.eos = self.getEoS(species, eos)

    def getEoS(self, components, eos):
        if eos == "PR":
            return PR(components)
        elif eos == "SRK":
            return SRK(components)
        elif eos == "AQ1":
            return AQ1(components)
        else:
            return AQ3(components)

    def component_parameters(self, p, T):
        self.eos.component_parameters(p, T)
        return

    def parameters(self, n, order=1):
        self.eos.parameters(n, order)
        return

    def setMolality(self, mi):
        self.eos.setMolality(mi)
        return

    def fugacityCoefficient(self, n):
        self.parameters(n)
        phi = np.zeros(self.NC)
        for i in range(self.NC):
            phi[i] = np.exp(self.eos.lnphii(i))
        return phi

    def gibbs(self, x, phi):
        return self.eos.gibbs(x, phi)


class EOSFlash(Flash):
    def __init__(self, components, ki, combined=False, mi=None):
        super().__init__(components, ki)

        self.ki = ki

        if mi is None:
            self.mi = []
        else:
            if combined:
                self.mi = np.array([mi[0], mi[0]])
            else:
                self.mi = mi

        self.f = np.zeros((2, 2))

        self.H2O_idx = components.index("H2O")
        self.CO2_idx = components.index("CO2")

        self.PR = EoS(["H2O", "CO2"], ["Na+", "Cl-"], "PR")
        self.AQ = EoS(["H2O", "CO2"], ["Na+", "Cl-"], "AQ3")

    def set_eos_parameters(self, p, T):
        self.PR.component_parameters(p, T)
        self.AQ.component_parameters(p, T)
        return

    def update_K(self, y, x, m):
        Y = np.array([y[self.H2O_idx], y[self.CO2_idx]])
        Y /= np.sum(Y)
        X = np.array([x[self.H2O_idx], x[self.CO2_idx]])
        X /= np.sum(X)

        if len(self.mi) != 0:
            self.AQ.setMolality(self.mi)
        else:
            self.AQ.setMolality(m)
        phi_V = self.PR.fugacityCoefficient(Y)
        phi_A = self.AQ.fugacityCoefficient(X)

        self.f[0, :] = Y[0:2] * phi_V
        self.f[1, :] = X[0:2] * phi_A

        self.K_values[self.H2O_idx] = phi_A[0]/phi_V[0]
        self.K_values[self.CO2_idx] = phi_A[1]/phi_V[1]
        # print(self.K_values)
        return

    def check_convergence(self):
        NP = 2
        tolerance = 1E-6
        for i in range(2):
            for j in range(NP - 1):
                for jj in range(j + 1, NP):
                    df = np.abs((self.f[j, i] - self.f[jj, i]) / self.f[j, i])
                    if df > tolerance:
                        return False
        return True


#  Density dependent on compressibility only
class Density4Ions:
    def __init__(self, density, compressibility=0, p_ref=1, ions_fac=0):
        super().__init__()
        # Density evaluator class based on simple first order compressibility approximation (Taylor expansion)
        self.density_rc = density
        self.cr = compressibility
        self.p_ref = p_ref
        self.ions_fac = ions_fac

    def evaluate(self, pres, ion_liq_molefrac):
        return self.density_rc * (1 + self.cr * (pres - self.p_ref) + self.ions_fac * ion_liq_molefrac)

class Density:
    def __init__(self, dens0=1000, compr=0, p0=1, x_mult=0):
        self.compr = compr
        self.p0 = p0
        self.dens0 = dens0
        self.x_max = x_mult

    def evaluate(self, pressure, x_co2):
        density = (self.dens0 + x_co2 * self.x_max) * (1 + self.compr * (pressure - self.p0))
        return density

class ViscosityConst:
    def __init__(self, visc):
        self.visc = visc

    def evaluate(self):
        return self.visc

class Enthalpy:
    def __init__(self, tref=273.15, hcap=0.0357):
        self.tref = tref
        self.hcap = hcap

    def evaluate(self, temp):
        # methane heat capacity
        enthalpy = self.hcap * (temp - self.tref)
        return enthalpy

class PhaseRelPerm:
    def __init__(self, phase, swc=0, sgr=0):
        self.phase = phase

        self.Swc = swc
        self.Sgr = sgr
        if phase == "oil":
            self.kre = 1
            self.sr = self.Swc
            self.sr1 = self.Sgr
            self.n = 2
        elif phase == 'gas':
            self.kre = 1
            self.sr = self.Sgr
            self.sr1 = self.Swc
            self.n = 2
        else:  # water
            self.kre = 1
            self.sr = self.Swc
            self.sr1 = self.Sgr
            self.n = 2


    def evaluate(self, sat):

        if sat >= 1 - self.sr1:
            kr = self.kre

        elif sat <= self.sr:
            kr = 0

        else:
            # general Brook-Corey
            kr = self.kre * ((sat - self.sr) / (1 - self.Sgr - self.Swc)) ** self.n

        return kr


class kinetic_basic():
    def __init__(self, equi_prod, kin_rate_cte, ne, combined_ions=True):
        self.equi_prod = equi_prod
        self.kin_rate_cte = kin_rate_cte
        self.kinetic_rate = np.zeros(ne)
        self.combined_ions = combined_ions

    def evaluate(self, x, nu_sol):
        self.kinetic_rate[:] = 0
        if self.combined_ions:
            ion_prod = (x[1][1] / 2) ** 2
            self.kinetic_rate[1] = - self.kin_rate_cte * (1 - ion_prod / self.equi_prod) * nu_sol
            self.kinetic_rate[-1] = - 0.5 * self.kinetic_rate[1]  # doesn't work when thermal is on!!!
        else:
            ion_prod = x[1][1] * x[1][2]
            self.kinetic_rate[1] = - self.kin_rate_cte * (1 - ion_prod / self.equi_prod) * nu_sol
            self.kinetic_rate[2] = - self.kin_rate_cte * (1 - ion_prod / self.equi_prod) * nu_sol
            self.kinetic_rate[-1] = - self.kinetic_rate[1]  # doesn't work when thermal is on!!!

        return self.kinetic_rate


class kinetic_basic2():
    def __init__(self, ion_idx, equi_prod, kin_rate_cte, ne, combined_ions=True):
        self.ion_idx = ion_idx
        self.solid_idx = -1

        self.equi_prod = equi_prod
        self.kin_rate_cte = kin_rate_cte
        self.kinetic_rate = np.zeros(ne)
        self.combined_ions = combined_ions

    def evaluate(self, x, nu_sol):
        self.kinetic_rate[:] = 0
        if self.combined_ions:
            if x[1][self.ion_idx] > x[0][self.ion_idx]:  # Aq phase present
                ion_prod = (x[1][self.ion_idx] / 2) ** 2
                const = self.kin_rate_cte
            else:
                ion_prod = (x[0][self.ion_idx] / 2) ** 2
                # ion_prod = 0.21
                const = self.kin_rate_cte
            # ion_prod = (x[1][self.ion_idx] / 2) ** 2
            self.kinetic_rate[self.ion_idx] = - const * (1 - ion_prod / self.equi_prod) * nu_sol
            self.kinetic_rate[-1] = - 0.5 * self.kinetic_rate[self.ion_idx]  # doesn't work when thermal is on!!!
        else:
            ion_prod = x[1][self.ion_idx[0]] * x[1][self.ion_idx[1]]
            self.kinetic_rate[self.ion_idx[0]] = - self.kin_rate_cte * (1 - ion_prod / self.equi_prod) * nu_sol
            self.kinetic_rate[self.ion_idx[1]] = - self.kin_rate_cte * (1 - ion_prod / self.equi_prod) * nu_sol
            self.kinetic_rate[-1] = - self.kinetic_rate[self.ion_idx[0]]  # doesn't work when thermal is on!!!

        return self.kinetic_rate