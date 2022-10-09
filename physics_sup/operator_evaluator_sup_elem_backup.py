import numpy as np
from darts.engines import *
from darts.physics import *

import os.path as osp

physics_name = osp.splitext(osp.basename(__file__))[0]

# Define our own operator evaluator class
class ReservoirOperators(operator_set_evaluator_iface):
    def __init__(self, property_container, thermal=0):
        super().__init__()  # Initialize base-class
        # Store your input parameters in self here, and initialize other parameters here in self
        self.min_z = property_container.min_z
        self.property = property_container
        self.thermal = thermal

        self.nc = self.property.nc
        self.nph = self.property.nph
        self.nm = self.property.nm
        self.nc_fl = self.nc - self.nm
        self.ne = self.property.nelem + self.thermal

        #                al  +          bt        +   gm     +   dlt   +         chi         + rock_temp por  + gr/cap  + por
        self.total = self.ne + self.ne * self.nph + self.nph + self.ne + self.ne * self.nph + 3 + 2 * self.nph + 1

        # Allocate memory for operators
        self.num_comp = self.property.rate_ann_mat.shape[1]
        self.num_elem = self.property.rate_ann_mat.shape[0]

        self.acc_vec = np.zeros((self.num_comp,))
        self.flux_vec_np = np.zeros((self.num_comp, self.nph))
        self.diff_vec_np = np.zeros((self.num_comp, self.nph))

    def evaluate(self, state, values):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        # Composition vector and pressure from state:
        vec_state_as_np = np.asarray(state)
        pressure = vec_state_as_np[0]

        self.acc_vec[:] = 0
        self.flux_vec_np[:] = 0
        self.diff_vec_np[:] = 0

        nc = self.property.nc
        nph = self.property.nph
        nm = self.property.nm
        nc_fl = nc - nm
        ne = self.property.nelem + self.thermal  # needs to be .nelem instead of nc

        #       al + bt        + gm + dlt + chi     + rock_temp por    + gr/cap  + por
        total = ne + ne * nph + nph + ne + ne * nph + 3 + 2 * nph + 1

        for i in range(total):
            values[i] = 0

        # numpy wrapper around to enable slicing:
        operator_array = np.array(values, copy=False)

        #  some arrays will be reused in thermal
        (self.sat, self.x, rho, self.rho_m, self.mu, self.kr, self.pc, self.ph) = self.property.evaluate(state)

        self.compr = (1 + self.property.rock_comp * (pressure - self.property.p_ref))  # compressible rock

        # Traditional:
        # density_tot = np.sum(self.sat * self.rho_m)
        # zc = np.append(vec_state_as_np[1:nc], 1 - np.sum(vec_state_as_np[1:nc]))
        # phi = 1 - np.sum(zc[nc_fl:nc])

        # Total composition:
        sol_frac = self.property.sol_phase_frac
        # sol_frac = 1 - np.sum(state[1:])
        nu_t = np.zeros((3,))
        nu_t[:-1] = self.property.nu * (1 - sol_frac)
        nu_t[-1] = sol_frac
        rho_m_all = np.zeros((3,))
        rho_m_all[:-1] = self.rho_m
        rho_m_all[-1] = self.property.solid_dens[0]
        zc_t = np.zeros((4,))
        zc_t[:-1] = self.x[0, :-1] * nu_t[0] + self.x[1, :-1] * nu_t[1]
        zc_t[-1] = 1 * nu_t[-1]
        total_density = 1 / (np.sum(nu_t[rho_m_all>0] / rho_m_all[rho_m_all>0]))
        total_elem_density = np.sum(np.dot(self.property.rate_ann_mat, zc_t)) / \
                             (np.sum(nu_t[rho_m_all>0] / rho_m_all[rho_m_all>0]))
        ze_test = (total_density / total_elem_density) * np.dot(self.property.rate_ann_mat, zc_t)

        ze = np.zeros((self.property.nelem,))
        ze[:-1] = vec_state_as_np[1:ne]
        ze[-1] = 1 - np.sum(ze)

        sat_all = (nu_t[rho_m_all>0] / rho_m_all[rho_m_all>0]) / (np.sum(nu_t[rho_m_all>0] / rho_m_all[rho_m_all>0]))
        phi = 1 - sat_all[-1]

        """ CONSTRUCT OPERATORS HERE """
        """ Alpha operator represents accumulation term """
        # for i in range(nc_fl):
        #     self.acc_vector[i] = self.compr * density_tot * zc[i]
        #
        # """ and alpha for mineral components """
        # for i in range(nm):
        #     self.acc_vector[i + nc_fl] = self.property.solid_dens[i] * zc[i + nc_fl]
        #
        # operator_array[:ne] = np.dot(self.property.flash_ev.rate_anni_mat, self.acc_vector)
        for i in range(ne):
            operator_array[i] = total_elem_density * ze[i]

        """ Beta operator represents flux term: """
        for j in self.ph:
            for i in range(nc_fl):
                self.flux_vec_np[i, j] = self.x[j][i] * self.rho_m[j] * self.kr[j] / self.mu[j]

        for ii in range(self.nph):
            operator_array[ne * (1 + ii):(2 + ii) * ne] = np.dot(self.property.rate_ann_mat, self.flux_vec_np[:, ii])

        """ Gamma operator for diffusion (same for thermal and isothermal) """
        shift = ne + ne * nph
        for j in self.ph:
            values[shift + j] = self.compr * self.sat[j]

        """ Chi operator for diffusion """
        shift += nph
        for i in range(nc):
            for j in self.ph:
                values[shift + i * nph + j] = self.property.diff_coef * self.x[j][i] * self.rho_m[j]
                self.diff_vec_np[i, j] = self.property.diff_coef * self.x[j][i] * self.rho_m[j]

        slice_vec = np.arange(0, self.nph * self.ne - 1, 2) + shift
        for ii in range(self.nph):
            operator_array[slice_vec + ii] = np.dot(self.property.rate_ann_mat, self.diff_vec_np[:, ii])

        """ Delta operator for reaction """
        shift += nph * ne
        if self.property.kinetic_rate_ev:
            kinetic_rate = self.property.kinetic_rate_ev.evaluate(self.x, 1 - phi)
            for i in range(ne):
                values[shift + i] = kinetic_rate[i]

        """ Gravity and Capillarity operators """
        shift += ne
        # E3-> gravity
        for i in self.ph:
            values[shift + 3 + i] = rho[i]

        # E4-> capillarity
        for i in self.ph:
            values[shift + 3 + nph + i] = self.pc[i]
        # E5_> porosity
        values[shift + 3 + 2 * nph] = phi

        #print(state, values)

        return 0

class WellOperators(operator_set_evaluator_iface):
    def __init__(self, property_container, thermal=0):
        super().__init__()  # Initialize base-class
        # Store your input parameters in self here, and initialize other parameters here in self
        self.nc = property_container.nc
        self.nph = property_container.nph
        self.min_z = property_container.min_z
        self.property = property_container
        self.thermal = thermal

        # Allocate memory for operators
        self.num_comp = self.property.rate_ann_mat.shape[1]
        self.num_elem = self.property.rate_ann_mat.shape[0]

        self.acc_vec = np.zeros((self.num_comp,))
        self.flux_vec_np = np.zeros((self.num_comp, self.nph))
        self.diff_vec_np = np.zeros((self.num_comp, self.nph))

    def evaluate(self, state, values):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        # Composition vector and pressure from state:
        vec_state_as_np = np.asarray(state)
        pressure = vec_state_as_np[0]

        self.acc_vec[:] = 0
        self.flux_vec_np[:] = 0
        self.diff_vec_np[:] = 0

        nc = self.property.nc
        nph = self.property.nph
        nm = self.property.nm
        nc_fl = nc - nm
        ne = self.property.nelem + self.thermal

        #       al + bt        + gm + dlt + chi     + rock_temp por    + gr/cap  + por
        total = ne + ne * nph + nph + ne + ne * nph + 3 + 2 * nph + 1

        for i in range(total):
            values[i] = 0

        (sat, x, rho, rho_m, mu, kr, pc, ph) = self.property.evaluate(state)

        self.compr = (1 + self.property.rock_comp * (pressure - self.property.p_ref))  # compressible rock

        density_tot = np.sum(sat * rho_m)
        zc = np.append(vec_state_as_np[1:nc], 1 - np.sum(vec_state_as_np[1:nc]))
        phi = 1

        sol_frac = self.property.sol_phase_frac
        # sol_frac = 1 - np.sum(state[1:])
        nu_t = np.zeros((3,))
        nu_t[:-1] = self.property.nu * (1 - sol_frac)
        nu_t[-1] = sol_frac
        rho_m_all = np.zeros((3,))
        rho_m_all[:-1] = rho_m
        rho_m_all[-1] = self.property.solid_dens[0]
        zc_t = np.zeros((4,))
        zc_t[:-1] = x[0, :-1] * nu_t[0] + x[1, :-1] * nu_t[1]
        zc_t[-1] = 1 * nu_t[-1]
        total_density = 1 / (np.sum(nu_t[rho_m_all > 0] / rho_m_all[rho_m_all > 0]))
        total_elem_density = np.sum(np.dot(self.property.rate_ann_mat, zc_t)) / \
                             (np.sum(nu_t[rho_m_all > 0] / rho_m_all[rho_m_all > 0]))
        ze_test = (total_density / total_elem_density) * np.dot(self.property.rate_ann_mat, zc_t)

        ze = np.zeros((self.property.nelem,))
        ze[:-1] = vec_state_as_np[1:ne]
        ze[-1] = 1 - np.sum(ze)

        """ CONSTRUCT OPERATORS HERE """
        # numpy wrapper around to enable slicing:
        operator_array = np.array(values, copy=False)

        """ Alpha operator represents accumulation term """
        # for i in range(nc_fl):
        #     values[i] = self.compr * density_tot * zc[i]

        # """ and alpha for mineral components """
        # for i in range(nm):
        #     values[i + nc_fl] = self.property.solid_dens[i] * zc[i + nc_fl]

        for i in range(ne):
            operator_array[i] = total_elem_density * ze[i]

        """ Beta operator represents flux term: """
        # for j in ph:
        #     shift = ne + ne * j
        #     for i in range(nc):
        #         values[shift + i] = x[j][i] * rho_m[j] * sat[j] / mu[j]

        for j in ph:
            for i in range(nc_fl):
                self.flux_vec_np[i, j] = x[j][i] * rho_m[j] * sat[j] / mu[j]

        for ii in range(self.nph):
            operator_array[ne * (1 + ii):(2 + ii) * ne] = np.dot(self.property.rate_ann_mat, self.flux_vec_np[:, ii])

        """ Gamma operator for diffusion (same for thermal and isothermal) """
        shift = ne + ne * nph

        """ Chi operator for diffusion """
        shift += nph

        """ Delta operator for reaction """
        shift += nph * ne
        if self.property.kinetic_rate_ev:
            kinetic_rate = self.property.kinetic_rate_ev.evaluate(x, 1e-12)
            for i in range(ne):
                values[shift + i] = kinetic_rate[i]

        """ Gravity and Capillarity operators """
        shift += ne
        # E3-> gravity
        for i in range(nph):
            values[shift + 3 + i] = rho[i]

        # E5_> porosity
        values[shift + 3 + 2 * nph] = phi

        #print(state, values)
        return 0

class RateOperators(operator_set_evaluator_iface):
    def __init__(self, property_container):
        super().__init__()  # Initialize base-class
        # Store your input parameters in self here, and initialize other parameters here in self
        self.nc = property_container.nc
        self.nph = property_container.nph
        self.min_z = property_container.min_z
        self.property = property_container
        self.flux = np.zeros(self.nc)

    def evaluate(self, state, values):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        for i in range(self.nph):
            values[i] = 0

        (sat, x, rho, rho_m, mu, kr, pc, ph) = self.property.evaluate(state)


        self.flux[:] = 0
        # step-1
        for j in ph:
            for i in range(self.nc):
                self.flux[i] += rho_m[j] * kr[j] * x[j][i] / mu[j]
        # step-2
        flux_sum = np.sum(self.flux)

        #(sat_sc, rho_m_sc) = self.property.evaluate_at_cond(1, self.flux/flux_sum)
        sat_sc = sat
        rho_m_sc = rho_m

        # step-3
        total_density = np.sum(sat_sc * rho_m_sc)
        # step-4
        for j in ph:
            values[j] = sat_sc[j] * flux_sum / total_density

        #print(state, values)
        return 0


# Define our own operator evaluator class
class ReservoirThermalOperators(ReservoirOperators):
    def __init__(self, property_container, thermal=1):
        super().__init__(property_container, thermal=thermal)  # Initialize base-class

    def evaluate(self, state, values):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        # Composition vector and pressure from state:
        super().evaluate(state, values)

        vec_state_as_np = np.asarray(state)
        pressure = state[0]
        temperature = vec_state_as_np[-1]

        # (enthalpy, rock_energy) = self.property.evaluate_thermal(state)
        (enthalpy, cond, rock_energy) = self.property.evaluate_thermal(state)

        nc = self.property.nc
        nph = self.property.nph
        ne = nc + self.thermal

        i = nc  # use this numeration for energy operators
        """ Alpha operator represents accumulation term: """
        for m in self.ph:
            values[i] += self.compr * self.sat[m] * self.rho_m[m] * enthalpy[m]  # fluid enthalpy (kJ/m3)
        values[i] -= self.compr * 100 * pressure

        """ Beta operator represents flux term: """
        for j in self.ph:
            shift = ne + ne * j
            values[shift + i] = enthalpy[j] * self.rho_m[j] * self.kr[j] / self.mu[j]

        """ Chi operator for temperature in conduction, gamma operators are skipped """
        shift = ne + ne * nph + nph
        for j in range(nph):
            # values[shift + nc * nph + j] = temperature
            values[shift + ne * j + nc] = temperature * cond[j]

        """ Delta operator for reaction """
        shift += nph * ne
        values[shift + i] = 0

        """ Additional energy operators """
        shift += ne
        # E1-> rock internal energy
        values[shift] = rock_energy / self.compr  # kJ/m3
        # E2-> rock temperature
        values[shift + 1] = temperature
        # E3-> rock conduction
        values[shift + 2] = 1 / self.compr  # kJ/m3

        #print(state, values)

        return 0


class PropertyEvaluator(operator_set_evaluator_iface):
    def __init__(self, property_container, thermal=0):
        super().__init__()  # Initialize base-class
        # Store your input parameters in self here, and initialize other parameters here in self
        self.min_z = property_container.min_z
        self.property = property_container
        self.thermal = thermal
        self.n_ops = 2 * self.property.nph

    def evaluate(self, state, values):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        # Composition vector and pressure from state:

        nph = self.property.nph


        #  some arrays will be reused in thermal
        (self.sat, self.x, rho, self.rho_m, self.mu, self.kr, self.pc, self.ph) = self.property.evaluate(state)

        for i in range(nph):
            values[i] = self.sat[i]

        for i in range(nph):
            values[i + nph] = self.rho_m[i]

        return
