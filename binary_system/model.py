from darts.models.reservoirs.struct_reservoir import StructReservoir
from darts.models.darts_model import DartsModel
from darts.engines import sim_params, value_vector
from darts.physics import *
import numpy as np
from properties_basic import *
from property_container import *
from darts.models.physics.iapws.iapws_property import *
from darts.models.physics.iapws.custom_rock_property import *
from physics_comp_sup import Compositional

import matplotlib.pyplot as plt


# Model class creation here!
class Model(DartsModel):
    def __init__(self):
        # Call base class constructor
        super().__init__()

        # Measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        self.zero = 1e-12
        init_ions = 0.5
        solid_init = 0.0
        equi_prod = (init_ions / 2) ** 2
        solid_inject = self.zero
        trans_exp = 3
        self.init_pres = 100
        self.inj_pres = 150
        self.prod_pres = 50

        """Reservoir"""
        perm = 100 / (1 - solid_init) ** trans_exp
        (self.nx, self.ny) = (1000, 1)
        self.reservoir = StructReservoir(self.timer, nx=self.nx, ny=1, nz=1, dx=1, dy=1, dz=1, permx=perm,
                                         permy=perm, permz=perm/10, poro=0.3, depth=1000)

        """well location"""
        self.reservoir.add_well("INJ_GAS")
        # self.reservoir.add_well("INJ_WAT")
        self.reservoir.add_perforation(well=self.reservoir.wells[-1], i=1, j=1, k=1, multi_segment=False)

        self.reservoir.add_well("PROD")
        self.reservoir.add_perforation(well=self.reservoir.wells[-1], i=self.nx, j=1, k=1, multi_segment=False)

        self.ini_comp = [self.zero]
        self.map = []

        """Physical properties"""
        # Create property containers:
        components_name = ['CO2', 'H2O']
        Mw = [44.01, 18.015]

        self.thermal = 0
        if self.thermal:
            hcap = np.array(self.reservoir.mesh.heat_capacity, copy=False)
            rcond = np.array(self.reservoir.mesh.rock_cond, copy=False)
            hcap.fill(2200)
            rcond.fill(181.44)

        self.property_container = model_properties(phases_name=['gas', 'wat'],
                                                   components_name=components_name, rock_comp=1e-7, Mw=Mw,
                                                   min_z=self.zero / 10, diff_coef=0e-9 * 60 * 60 * 24)

        self.components = self.property_container.components_name
        self.phases = self.property_container.phases_name
        self.property_container.salinity = 1

        """ properties correlations """
        self.property_container.flash_ev = Flash(self.components, [10, 1e-1], self.zero)


        self.property_container.density_ev = dict([('gas', Density(compr=1e-4, dens0=100)),
                                                   ('wat', Density(compr=1e-6, dens0=1000))])
        self.property_container.viscosity_ev = dict([('gas', ViscosityConst(0.1)),
                                                     ('wat', ViscosityConst(1))])
        self.property_container.rel_perm_ev = dict([('gas', PhaseRelPerm("gas")),
                                                    ('wat', PhaseRelPerm("wat"))])

        self.property_container.enthalpy_ev = dict([('gas', GasEnthalpy()),
                                                    ('wat', AqEnthalpy(self.property_container.salinity))])
        self.property_container.conductivity_ev = dict([('gas', GasCondonductivity()),
                                                        ('wat', AqCondonductivity(self.property_container.salinity))])
        self.property_container.rock_energy_ev = RockEnergyEvaluator()

        ne = self.property_container.nc + self.thermal
        # self.property_container.kinetic_rate_ev = kinetic_basic(equi_prod, 1e-0, ne)

        output_props = PropertyEvaluator(self.property_container, ne, self.thermal)
        # output_props = PropertyEvaluator
        """ Activate physics """
        self.physics = Compositional(self.property_container, self.timer, n_points=64, min_p=1, max_p=1000,
                                     min_z=self.zero/10, max_z=1-self.zero/10, cache=0, out_props=output_props,
                                     thermal=self.thermal, min_t=293, max_t=393)

        self.inj_stream_gas = [1 - 2 * self.zero]
        self.inj_stream_wat = [2 * self.zero]

        if self.thermal:
            self.inj_stream_gas += [300]
            self.inj_stream_wat += [300]

        self.params.trans_mult_exp = trans_exp
        # Some newton parameters for non-linear solution:
        self.params.first_ts = 1e-6
        self.params.max_ts = 0.1
        self.params.mult_ts = 2

        self.params.tolerance_newton = 1e-4
        self.params.tolerance_linear = 1e-5
        self.params.max_i_newton = 10
        self.params.max_i_linear = 50
        self.params.newton_type = sim_params.newton_local_chop
        # self.params.newton_params[0] = 0.2

        self.timer.node["initialization"].stop()

    # Initialize reservoir and set boundary conditions:
    def set_initial_conditions(self):
        """ initialize conditions for all scenarios"""
        if self.thermal:
            self.physics.set_uniform_T_initial_conditions(self.reservoir.mesh, uniform_pressure=self.init_pres,
                                                          uniform_composition=self.ini_comp, uniform_temp=348.15)
        else:
            self.physics.set_uniform_initial_conditions(self.reservoir.mesh, self.init_pres, self.ini_comp)


        return

    def set_boundary_conditions(self):
        for i, w in enumerate(self.reservoir.wells):
            if "INJ_GAS" in w.name:
                # w.control = self.physics.new_rate_inj(self.inj_gas_rate, self.inj_stream_gas, 0)
                w.control = self.physics.new_bhp_inj(self.inj_pres, self.inj_stream_gas)
            elif "INJ_WAT" in w.name:
                # w.control = self.physics.new_rate_inj(self.inj_wat_rate, self.inj_stream_wat, 1)
                w.control = self.physics.new_bhp_inj(self.inj_pres, self.inj_stream_wat)
            else:
                w.control = self.physics.new_bhp_prod(self.prod_pres)

    def set_op_list(self):
        self.op_num = np.array(self.reservoir.mesh.op_num, copy=False)
        n_res = self.reservoir.mesh.n_res_blocks
        self.op_num[n_res:] = 1
        self.op_list = [self.physics.acc_flux_itor, self.physics.acc_flux_w_itor]

    def properties(self, state):
        (sat, x, rho, rho_m, mu, kr, ph) = self.property_container.evaluate(state)
        return sat[0]

    def print_and_plot_1D(self, filename):
        nc = self.property_container.nc
        Sg = np.zeros(self.reservoir.nb)

        Xn = np.array(self.physics.engine.X, copy=True)
        P = Xn[0:self.reservoir.nb * nc:nc]
        z_co2 = Xn[1:self.reservoir.nb * nc:nc]
        z_h2o = 1 - z_co2

        for ii in range(self.reservoir.nb):
            x_list = Xn[ii*nc:(ii+1)*nc]
            state = value_vector(x_list)
            (sat, x, rho, rho_m, mu, kr, pc, ph) = self.property_container.evaluate(state)
            Sg[ii] = sat[0]

        """ start plots """

        font_dict_title = {'family': 'sans-serif',
                           'color': 'black',
                           'weight': 'normal',
                           'size': 14,
                           }

        font_dict_axes = {'family': 'monospace',
                          'color': 'black',
                          'weight': 'normal',
                          'size': 14,
                          }

        fig, axs = plt.subplots(2, 2, figsize=(12, 10), dpi=200, facecolor='w', edgecolor='k')
        """ sg and x """
        axs[0][0].plot(z_co2, 'b')
        axs[0][0].set_xlabel('x [m]', font_dict_axes)
        axs[0][0].set_ylabel('$z_{CO_2}$ [-]', font_dict_axes)
        axs[0][0].set_title('Fluid composition', fontdict=font_dict_title)

        axs[0][1].plot(z_h2o, 'b')
        axs[0][1].set_xlabel('x [m]', font_dict_axes)
        axs[0][1].set_ylabel('$z_{H_2O}$ [-]', font_dict_axes)
        axs[0][1].set_title('Fluid composition', fontdict=font_dict_title)

        axs[1][0].plot(P, 'b')
        axs[1][0].set_xlabel('x [m]', font_dict_axes)
        axs[1][0].set_ylabel('$p$ [bar]', font_dict_axes)
        axs[1][0].set_title('Pressure', fontdict=font_dict_title)

        axs[1][1].plot(Sg, 'b')
        axs[1][1].set_xlabel('x [m]', font_dict_axes)
        axs[1][1].set_ylabel('$s_g$ [-]', font_dict_axes)
        axs[1][1].set_title('Gas saturation', fontdict=font_dict_title)

        left = 0.05  # the left side of the subplots of the figure
        right = 0.95  # the right side of the subplots of the figure
        bottom = 0.05  # the bottom of the subplots of the figure
        top = 0.95  # the top of the subplots of the figure
        wspace = 0.25  # the amount of width reserved for blank space between subplots
        hspace = 0.25  # the amount of height reserved for white space between subplots
        plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

        for ii in range(2):
            for jj in range(2):
                for tick in axs[ii][jj].xaxis.get_major_ticks():
                    tick.label.set_fontsize(20)

                for tick in axs[ii][jj].yaxis.get_major_ticks():
                    tick.label.set_fontsize(20)

        plt.tight_layout()
        plt.savefig(filename)
        plt.show()

        # Write to numpy array:
        output_data = np.zeros((self.reservoir.nb, 4))
        output_data[:, 0] = z_co2
        output_data[:, 1] = z_h2o
        output_data[:, 2] = P
        output_data[:, 3] = Sg
        np.save(filename[:-4], output_data)


class model_properties(property_container):
    def __init__(self, phases_name, components_name, Mw, min_z=1e-11,
                 diff_coef=0.0, rock_comp=1e-6, solid_dens=None):
        # Call base class constructor
        # Cm = 0
        # super().__init__(phases_name, components_name, Mw, Cm, min_z, diff_coef, rock_comp, solid_dens)
        if solid_dens is None:
            solid_dens = []
        super().__init__(phases_name, components_name, Mw, min_z=min_z, diff_coef=diff_coef,
                         rock_comp=rock_comp, solid_dens=solid_dens)

    def run_flash(self, pressure, zc):

        nc_fl = self.nc - self.nm
        norm = 1 - np.sum(zc[nc_fl:])

        zc_r = zc[:nc_fl] / norm
        (xr, nu) = self.flash_ev.evaluate(pressure, zc_r)
        V = nu[0]

        if V <= 0:
            V = 0
            xr[1] = zc_r
            ph = [1]
        elif V >= 1:
            V = 1
            xr[0] = zc_r
            ph = [0]
        else:
            ph = [0, 1]

        for i in range(nc_fl):
            for j in range(2):
                self.x[j][i] = xr[j][i]

        self.nu[0] = V
        self.nu[1] = (1 - V)

        return ph

class PropertyEvaluator(operator_set_evaluator_iface):
    def __init__(self, property_container, ne, thermal=0):
        super().__init__()  # Initialize base-class
        # Store your input parameters in self here, and initialize other parameters here in self
        self.min_z = property_container.min_z
        self.property = property_container
        self.thermal = thermal
        # self.n_ops = 4 * self.property.nph
        self.n_ops = ne * 2

    def evaluate(self, state, values):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        # Composition vector and pressure from state:

        #  some arrays will be reused in thermal
        (sat, x, rho, rho_m, mu, kr, pc, ph) = self.property.evaluate(state)

        nph = self.property.nph
        for i in range(nph):
            values[i + 0 * nph] = sat[i]
            values[i + 1 * nph] = rho[i]
            values[i + 2 * nph] = rho[i]
            values[i + 3 * nph] = kr[i]

        return 0

class GasEnthalpy(property_evaluator_iface):
    def __init__(self):
        super().__init__()
        #Values for Guo
        self.Tref = 1000  # Kelvin
        self.R = 8.3145
        # Calculate
        # For CH4,CO2, use Yongfan Guo
        # parameters for ideal gas enthalpy for CO2, H2O
        self.a_ideal = np.array([[-1.8188731,12.903022,-9.6634864,4.2251879,-1.042164,0.12683515,-0.49939675, 2.4950242,
                                  -0.8272375,0.15372481,-0.015861243,0.000860172,1.92222E-05, 8191.6797272+9365],
                                 [31.04096012,-39.14220805,37.96952772,-21.8374911,7.422514946,-1.381789296,0.108807068,
                                  -12.07711768,3.391050789,-0.58452098,0.058993085,-0.0031297,6.57461*1e-5, 9908-113781.35]])

    def evaluate(self, state, y, H_dev):
        T = state[2]
        dH_vap_H2O = 0
        enthalpy = self.ideal_Guo(y, T) + H_dev
        return enthalpy[0]/1000-dH_vap_H2O

    def ideal_Guo(self, y, T):
        a_mix = np.zeros(14)
        for ii in range(0, 14):
            a_mix[ii] = self.a_ideal[1,ii]*y[1] + self.a_ideal[0,ii]*y[0]
        tau = T/self.Tref
        a_temp = 0
        for ii in range(0,13):
            if ii < 7:
                a_temp = a_temp + a_mix[ii]/(ii+1)*tau**(ii+1)
            elif ii == 7:
                a_temp = a_temp + a_mix[ii]*np.log(tau)
            elif ii<13:
                a_temp = a_temp + a_mix[ii]/(7-ii)*(1/tau)**(ii-7)

        H_ideal = self.R*self.Tref*a_temp + a_mix[-1]

        return H_ideal

######################################################################################################
# Aqueous phase enthalpy
class AqEnthalpy(property_evaluator_iface):  # Assume pure h20 for now
    def __init__(self, Cm):
        super().__init__()
        self.Cm = Cm

    def evaluate(self, state, x):
        vec_state_as_np = np.asarray(state)
        P = vec_state_as_np[0]
        T = vec_state_as_np[-1]
        Tc = T - 273.15

        Hw, Hs = self.Hpure(Tc)
        # print(Hw,Hs)
        if Tc < 150:
            Hsb = self.Michalides(Hw, Hs, Tc)
        # elif 99.9 <= Tc <= 130.1:
        #     Hsb1 = self.Michalides(Hw, Hs, Tc)
        #     Hsb2 = self.Lorenz(Hw, Hs, Tc)
        #     Hsb = (Hsb1 + Hsb2) / 2
        # elif 130.1 < Tc <= 300:
        #     Hsb = self.Lorenz(Hw, Hs, Tc)
        else:
            print('T out of bounds for Aq enthalpy', T)

        # print(Hsb)
        dH_diss_g = self.dH_diss_gas(P, T)
        Hbrine = self.Hrevised(P, T, Hsb)
        # print(dH_diss_g)
        return Hbrine - dH_diss_g * x[0]

    def rho_brine(self, P, Tc):  # Pressure, Temperature, Molality

        # ref; spivey et al. 2004
        P0 = 700  # 70 MPa reference pressure

        # TEMP IN CELCIUS
        # PRESSURE IN BAR
        # SALINITY IN MOLES/KG
        def eq_3(a1, a2, a3, a4, a5, Tc):
            a = (a1 * (Tc / 100) ** 2 + a2 * (Tc / 100) + a3) / (a4 * (Tc / 100) ** 2 + a5 * (Tc / 100) + 1)
            return a

        # EoS coeff. for pure water
        Dw = np.array([-0.127213, 0.645486, 1.03265, -0.070291, 0.639589])
        E_w = np.array([4.221, -3.478, 6.221, 0.5182, -0.4405])
        F_w = np.array([-11.403, 29.932, 27.952, 0.20684, 0.3768])
        # coeff. for density of brine at ref. pressure 70 MPa as a function of Temp.
        D_cm2 = np.array([-7.925e-5, -1.93e-6, -3.4254e-4, 0, 0])
        D_cm32 = np.array([1.0998e-3, -2.8755e-3, -3.5819e-3, -0.72877, 1.92016])
        D_cm1 = np.array([-7.6402e-3, 3.6963e-2, 4.36083e-2, -0.333661, 1.185685])
        D_cm12 = np.array([3.746e-4, -3.328e-4, -3.346e-4, 0, 0])
        # Coeff. for eq. 12 & 13 for brine compressibility
        E_cm = np.array([0, 0, 0.1353, 0, 0])
        F_cm32 = np.array([-1.409, -0.361, -0.2532, 0, 9.216])
        F_cm1 = np.array([0, 5.614, 4.6782, -0.307, 2.6069])
        F_cm12 = np.array([-0.1127, 0.2047, -0.0452, 0, 0])

        # Equation 3=> function of temp.
        Dcm2 = eq_3(D_cm2[0], D_cm2[1], D_cm2[2], D_cm2[3], D_cm2[4], Tc)
        Dcm32 = eq_3(D_cm32[0], D_cm32[1], D_cm32[2], D_cm32[3], D_cm32[4], Tc)
        Dcm1 = eq_3(D_cm1[0], D_cm1[1], D_cm1[2], D_cm1[3], D_cm1[4], Tc)
        Dcm12 = eq_3(D_cm12[0], D_cm12[1], D_cm12[2], D_cm12[3], D_cm12[4], Tc)

        rho_w0 = eq_3(Dw[0], Dw[1], Dw[2], Dw[3], Dw[4], Tc)  # density of water at 70Mpa
        # density of pure water
        Ew_w = eq_3(E_w[0], E_w[1], E_w[2], E_w[3], E_w[4], Tc)
        Fw_w = eq_3(F_w[0], F_w[1], F_w[2], F_w[3], F_w[4], Tc)
        Iw = (1 / Ew_w) * np.log(abs(Ew_w * (P / P0) + Fw_w))
        Iw0 = (1 / Ew_w) * np.log(abs(Ew_w * (P0 / P0) + Fw_w))

        rho_w = rho_w0 * np.exp(Iw - Iw0)
        # print('rho_w',rho_w)
        ## density of Brine
        Ew = eq_3(E_w[0], E_w[1], E_w[2], E_w[3], E_w[4], Tc)
        Fw = eq_3(F_w[0], F_w[1], F_w[2], F_w[3], F_w[4], Tc)
        Ecm = eq_3(E_cm[0], E_cm[1], E_cm[2], E_cm[3], E_cm[4], Tc)

        Fcm32 = eq_3(F_cm32[0], F_cm32[1], F_cm32[2], F_cm32[3], F_cm32[4], Tc)
        Fcm1 = eq_3(F_cm1[0], F_cm1[1], F_cm1[2], F_cm1[3], F_cm1[4], Tc)
        Fcm12 = eq_3(F_cm12[0], F_cm12[1], F_cm12[2], F_cm12[3], F_cm12[4], Tc)

        # function of Salinity Cm
        rho_b0 = rho_w0 + Dcm2 * self.Cm ** 2 + Dcm32 * self.Cm ** (3 / 2) + Dcm1 * self.Cm + Dcm12 * self.Cm ** (1 / 2)

        Eb = Ew + Ecm * self.Cm
        Fb = Fw + Fcm32 * self.Cm ** (3 / 2) + Fcm1 * self.Cm + Fcm12 * self.Cm ** (1 / 2)

        # function of pressure
        Ib_p = (1 / Eb) * np.log(abs(Eb * (P / P0) + Fb))
        Ib_p0 = (1 / Eb) * np.log(abs(Eb * (P0 / P0) + Fb))

        # density of brine
        rho_b = rho_b0 * np.exp(Ib_p - Ib_p0)

        return rho_b * 1e3, rho_w * 1e3

    # Pure water enthalpy and salt give good correlations, validated
    def Hpure(self, T):  # Keenan, Keyes, Hill and Moore
        Hw = 0.12453e-4 * T ** 3 - 0.4513e-2 * T ** 2 + 4.81155 * T - 29.578
        Hs = (-0.83624e-3 * T ** 3 + 0.16792 * T ** 2 - 25.9293 * T) * (4.184 / 58.44)
        return Hw, Hs  # in kJ/kg  /58.4428/55.55

    def Antoine(self, Tc):
        if Tc < 99.9:
            Psat = 0.00133322 * 10 ** (8.07131 - (1730.63 / (233.426 + Tc)))
        elif 99.9 <= Tc <= 100.1:
            Psat1 = 10 ** (8.07131 - (1730.63 / (233.426 + Tc)))
            Psat2 = 10 ** (8.140191 - (1810.94 / (244.485 + Tc)))
            Psat = 0.00133322 * (Psat1 + Psat2) / 2
        elif 100.1 < Tc <= 374:
            Psat = 0.00133322 * 10 ** (8.140191 - (1810.94 / (244.485 + Tc)))  # To bar
        else:
            print('T out of bounds for Antoine')
        return Psat

    def Michalides(self, Hw, Hs, T):
        # Input in Kj/Kg , mol/kg and Celsius
        aij = np.array([[-9633.6, -4080.0, 286.49],
                        [166.58, 68.577, -4.6856],
                        [-0.90963, -0.36524, 0.0249667],
                        [0.17965e-2, 0.71924e-3, -0.4900e-4]])

        dH_diss_salt = 0
        for ii in [0, 1, 2, 3]:
            for jj in [0, 1, 2]:
                dH_diss_salt += aij[ii, jj] * T ** ii * self.Cm ** jj
        dH_diss_salt *= 4.184 / (1000 + 58.44 * self.Cm)

        x1 = 1000 / (1000 + 58.44 * self.Cm)
        x2 = 58.44 * self.Cm / (1000 + 58.44 * self.Cm)
        return x1 * Hw + x2 * Hs + dH_diss_salt

    # Lorenz Function gives correct results -> Validated
    def Lorenz(self, Hw, Hs, Tc):

        # Input Kj/mol, Kj,mol, mol/kg, C
        # Needs wt% and C so multiply cm by 58.44
        # mol/kg to kg/kg % = 1100g/kg
        wtpct = self.Cm * 58.44 / (1000 + self.Cm * 58.44) * 100

        bij = np.array([[0.2985, -7.257e-2, 1.071e-3],
                        [-7.819e-2, 2.169e-3, -3.343e-5],
                        [3.479e-4, -1.809e-5, 3.450e-7],
                        [-1.203e-6, 5.910e-8, -1.131e-9]])

        dH_diss_salt = 0
        for jj in [1, 2, 3]:
            Ai_t = 0
            for ii in [0, 1, 2, 3]:
                Ai_t += bij[ii, jj - 1] * (Tc ** ii)
            dH_diss_salt += Ai_t * (wtpct ** jj)

        x1 = 1
        x2 = 0

        return x1 * Hw + x2 * Hs + dH_diss_salt

    def dH_diss_gas(self, P, T):
        T2 = T + 0.1  # Forward difference with 1 K
        R = 8.31446  # J/mol/K
        phi_c, Hcoeff1 = aqueous(P, T, ['CO2'], self.Cm)
        phi_c, Hcoeff2 = aqueous(P, T2, ['CO2'], self.Cm)

        dH_diss_g = (np.log(Hcoeff2) - np.log(Hcoeff1)) / (1 / T2 - 1 / T) * R / 44.01  # in Kj/Kg Co2

        return dH_diss_g / (1000 / 44.01)  # convert to Kj/mol

    def Hrevised(self, P, T, Hsb):
        # Input in Bar, Kelvin, mol/kg, kg/m3
        # print(rho_b)
        T_dt = 1
        T_forward = T + T_dt
        rho_b, rho_w = self.rho_brine(P, T - 273.15)
        rho_b_f, rho_w_f = self.rho_brine(P, T_forward - 273.15)
        V = 1 / rho_b
        # print(V)
        dVdT = (1 / rho_b_f - 1 / rho_b) / T_dt
        # print(dVdT)
        Psat = self.Antoine(T - 273.15)  # Antoine equation, outputs in bar
        # print(Psat)
        Interim = (V - (T - 273.15) * dVdT)
        Mw_brine = 1000 / (1000 / 18.01 + self.Cm) + 58.44 * self.Cm / (1000 / 18.01 + self.Cm)
        # print(Mw_brine,'g/mol')
        Hbrine = Hsb / (1000 / (Mw_brine)) + Interim * (P - Psat)
        Factor = 1  # 1000/18.015#(1000 / 18.015 + Cm)/(1000 + 58.44 * Cm)  #to kJ/mol
        # print('Kg/mol', Factor)
        Hbrine = Hbrine / Factor
        # print(Hbrine,'kJ/mol')
        return Hbrine


class AqCondonductivity(property_evaluator_iface):
    def __init__(self, Cm):
        super().__init__()
        self.S = Cm * 58.44

    def evaluate(self, state):
        T = state[2]
        T_d = T / 300
        cond_aq = 0.797015 * T_d ** -0.194 - 0.251242 * T_d ** -4.717 + 0.096437 * T_d ** -6.385 - 0.032696 * T_d ** -2.134
        cond_brine = (cond_aq / (0.00022 * self.S + 1)) +0.00005*(state[0]-50)
        return cond_brine/1000*3600*24 #Convert from W/m/k to kj/m/day/K

class GasCondonductivity(property_evaluator_iface):
    def __init__(self):
        super().__init__()
        self.A = [105.161, 0.9007, 0.0007, 3.5e-15, 3.76e-10, 0.75, 0.0017]

    def evaluate(self, rho_g, state, sg):
        T = state[2]
        A = self.A
        cond_g = (A[0] + A[1] * rho_g + A[2] * rho_g ** 2 + A[3] * rho_g ** 3 * T ** 3 + A[4] * rho_g ** 4 + A[
            5] * T + A[6] * T ** 2) / np.sqrt(T)
        return cond_g * 1e-3/1000*3600*24 #Convert from W/m/k to kj/m/day/K

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

    def evaluate(self, state):
        T = state[2]
        T_ref = 273.15
        # c_vr = 3710  # 1400 J/kg.K * 2650 kg/m3 -> kJ/m3
        c_vr =  1 # 1400 J/kg.K * 2650 kg/m3 -> kJ/m3

        return c_vr * (T - T_ref)  # kJ/m3
