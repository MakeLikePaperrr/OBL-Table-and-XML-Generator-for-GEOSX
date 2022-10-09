from darts.models.reservoirs.struct_reservoir import StructReservoir
from darts.models.darts_model import DartsModel
from darts.engines import sim_params, value_vector
from darts.physics import *
import numpy as np
from properties_basic import *
from property_container import *
from physics_comp_sup import Compositional
import matplotlib.pyplot as plt


def create_map(lx, ly, nx, ny):

    map = np.ones((nx, ny))

    x1 = 120
    y1 = 80

    DX = lx / nx
    DY = ly / ny

    nx1 = int(np.ceil(x1 / DX))
    nx2 = int(np.floor((lx - x1) / DX))
    ny1 = int(np.ceil(y1 / DY))
    ny2 = int(np.floor((ly - y1) / DY))

    map[nx1:nx2, ny1:ny2] = 0

    map = np.reshape(map, (nx * ny,), order='F')

    return map


# Model class creation here!
class Model(DartsModel):
    def __init__(self, grid_1D=True, obl_pts=32, min_z_vec=1e-13, max_z_vec=1-1e-13):
        # Call base class constructor
        super().__init__()
        self.grid_1D = grid_1D

        # Measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        self.zero = 1e-12
        init_ions = 0.5
        solid_init = 0.7
        equi_prod = (init_ions / 2) ** 2
        solid_inject = self.zero
        trans_exp = 3
        self.combined_ions = True
        self.init_pres = 100
        self.physics_type = 'kin'  # equi or kin

        """Reservoir"""
        if grid_1D:
            perm = 100 / (1 - solid_init) ** trans_exp
            (self.nx, self.ny) = (1000, 1)
            self.reservoir = StructReservoir(self.timer, nx=self.nx, ny=1, nz=1, dx=1, dy=1, dz=1, permx=perm,
                                             permy=perm, permz=perm/10, poro=1, depth=1000)

            """well location"""
            self.reservoir.add_well("INJ_GAS")
            self.reservoir.add_perforation(well=self.reservoir.wells[-1], i=1, j=1, k=1, multi_segment=False)

            self.reservoir.add_well("PROD")
            self.reservoir.add_perforation(well=self.reservoir.wells[-1], i=self.nx, j=1, k=1, multi_segment=False)

            self.inj_gas_rate = 0.2

            zc_fl_init = [self.zero / (1 - solid_init), init_ions]
            zc_fl_init = zc_fl_init + [1 - sum(zc_fl_init)]
            self.ini_comp = [x * (1 - solid_init) for x in zc_fl_init]

            self.map = []
        else:
            (Lx, Ly) = (600, 240)
            (self.nx, self.ny) = (120, 48)
            dx = Lx / self.nx
            dy = Ly / self.ny

            self.map = create_map(Lx, Ly, self.nx, self.ny)

            perm = np.ones(self.nx * self.ny) * 100 / (1 - solid_init) ** trans_exp

            # Add inclination in y-direction:
            self.depth = np.ones((self.nx * self.ny,)) * 1000
            for j in range(self.ny):
                self.depth[j * self.nx:(j + 1) * self.nx] += j * dy

            self.reservoir = StructReservoir(self.timer, nx=self.nx, ny=1, nz=self.ny, dx=dx, dy=10, dz=dy, permx=perm,
                                             permy=perm, permz=perm, poro=1, depth=self.depth)

            # Perforate the left and right boundary:
            for ii in range(int(self.ny / 2)):
                self.reservoir.add_well("INJ_WAT_" + str(ii + 1))
                self.reservoir.add_perforation(well=self.reservoir.wells[-1], i=1, j=1, k=ii + 1, multi_segment=False)

            for ii in range(int(self.ny / 2), self.ny):
                self.reservoir.add_well("INJ_GAS_" + str(ii + 1))
                self.reservoir.add_perforation(well=self.reservoir.wells[-1], i=1, j=1, k=ii + 1,
                                               multi_segment=False)

            for ii in range(self.ny):
                self.reservoir.add_well("PROD_" + str(ii + 1))
                self.reservoir.add_perforation(well=self.reservoir.wells[-1], i=self.nx, j=1, k=ii + 1,
                                               multi_segment=False)

            self.inj_gas_rate = 1000 / self.ny * 2
            self.inj_wat_rate = 200 / self.ny * 2

            solid_void = 0.2
            if self.combined_ions:
                zc_fl_init = [self.zero / (1 - solid_init), init_ions]
            else:
                # zc_fl_init = [self.zero / (1 - solid_init), init_ions, self.zero / (1 - solid_init)]
                zc_fl_init = [self.zero / (1 - solid_init), init_ions / 2, init_ions / 2]
            zc_fl_init = zc_fl_init + [1 - sum(zc_fl_init)]
            self.ini_comp = [x * (1 - solid_init) for x in zc_fl_init]
            self.ini_void = [x * (1 - solid_void) for x in zc_fl_init]

        """Physical properties"""
        # Create property containers:
        if self.combined_ions:
            components_name = ['CO2', 'Ions', 'H2O', 'CaCO3']
            Mw = [44.01, (40.078 + 60.008) / 2, 18.015, 100.086]
        else:
            components_name = ['CO2', 'Ca', 'CO3', 'H2O', 'CaCO3']
            Mw = [44.01, 40.078, 60.008, 18.015, 100.086]
            # Mw = [44.01, (40.078 + 60.008) / 2, (40.078 + 60.008) / 2, 18.015, 100.086]

        self.thermal = 0
        self.property_container = model_properties(phases_name=['gas', 'wat'],
                                                   components_name=components_name, rock_comp=1e-7, Mw=Mw,
                                                   min_z=self.zero / 10, diff_coef=1e-9 * 60 * 60 * 24,
                                                   solid_dens=[2000])

        self.components = self.property_container.components_name
        self.phases = self.property_container.phases_name

        """ properties correlations """
        if self.combined_ions:
            self.property_container.flash_ev = Flash(self.components[:-1], [10, 1e-12, 1e-1], self.zero)
        else:
            self.property_container.flash_ev = Flash(self.components[:-1], [10, 1e-12, 1e-12, 1e-1], self.zero)

        self.property_container.density_ev = dict([('gas', Density(compr=1e-4, dens0=100)),
                                                   ('wat', Density(compr=1e-6, dens0=1000))])
        self.property_container.viscosity_ev = dict([('gas', ViscosityConst(0.1)),
                                                     ('wat', ViscosityConst(1))])
        self.property_container.rel_perm_ev = dict([('gas', PhaseRelPerm("gas")),
                                                    ('wat', PhaseRelPerm("wat"))])


        ne = self.property_container.nc + self.thermal
        self.property_container.kinetic_rate_ev = kinetic_basic(equi_prod, 1e-0, ne, self.combined_ions)

        output_props = PropertyEvaluator(self.property_container, ne, self.thermal)
        # output_props = PropertyEvaluator
        """ Activate physics """
        # self.physics = Compositional(self.property_container, self.timer, n_points=obl_pts, min_p=25, max_p=500,
        #                              min_z=self.zero/10, max_z=1-self.zero/10, cache=0, out_props=output_props)
        self.physics = Compositional(self.property_container, self.timer, n_points=obl_pts, min_p=25, max_p=500,
                                     min_z=min_z_vec, max_z=max_z_vec, cache=0, out_props=output_props)
        if self.combined_ions:
            zc_fl_inj_stream_gas = [1 - 2 * self.zero / (1 - solid_inject), self.zero / (1 - solid_inject)]
            zc_fl_inj_stream_liq = [2 * self.zero / (1 - solid_inject), self.zero / (1 - solid_inject)]
        else:
            zc_fl_inj_stream_gas = [1 - 3 * self.zero / (1 - solid_inject), self.zero / (1 - solid_inject), self.zero / (1 - solid_inject)]
            zc_fl_inj_stream_liq = [3 * self.zero / (1 - solid_inject), self.zero / (1 - solid_inject),
                                    self.zero / (1 - solid_inject)]

        zc_fl_inj_stream_gas = zc_fl_inj_stream_gas + [1 - sum(zc_fl_inj_stream_gas)]
        self.inj_stream_gas = [x * (1 - solid_inject) for x in zc_fl_inj_stream_gas]

        zc_fl_inj_stream_liq = zc_fl_inj_stream_liq + [1 - sum(zc_fl_inj_stream_liq)]
        self.inj_stream_wat = [x * (1 - solid_inject) for x in zc_fl_inj_stream_liq]

        self.params.trans_mult_exp = trans_exp
        # Some newton parameters for non-linear solution:
        self.params.first_ts = 0.001
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
        self.physics.set_uniform_initial_conditions(self.reservoir.mesh, self.init_pres, self.ini_comp)

        if len(self.map) > 0:
            nc = self.property_container.nc
            nb = self.reservoir.nb
            composition = np.array(self.reservoir.mesh.composition, copy=False)
            zc = np.zeros(nb)
            for i in range(nc-1):
                zc[:] = self.ini_comp[i]
                zc[self.map == 0] = self.ini_void[i]
                composition[i:(nc-1)*nb:nc-1] = zc
        return

    def set_boundary_conditions(self):
        for i, w in enumerate(self.reservoir.wells):
            if "INJ_GAS" in w.name:
                # w.control = self.physics.new_rate_inj(self.inj_gas_rate, self.inj_stream_gas, 0)
                w.control = self.physics.new_bhp_inj(125, self.inj_stream_gas)
            elif "INJ_WAT" in w.name:
                # w.control = self.physics.new_rate_inj(self.inj_wat_rate, self.inj_stream_wat, 1)
                w.control = self.physics.new_bhp_inj(125, self.inj_stream_wat)
            else:
                w.control = self.physics.new_bhp_prod(75)

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
        Ss = np.zeros(self.reservoir.nb)
        X = np.zeros((self.reservoir.nb, nc - 1, 2))

        rel_perm = np.zeros((self.reservoir.nb, 2))
        visc = np.zeros((self.reservoir.nb, 2))
        density = np.zeros((self.reservoir.nb, 3))
        density_m = np.zeros((self.reservoir.nb, 3))

        Xn = np.array(self.physics.engine.X, copy=True)

        P = Xn[0:self.reservoir.nb * nc:nc]
        z_caco3 = 1 - (Xn[1:self.reservoir.nb * nc:nc] + Xn[2:self.reservoir.nb * nc:nc] + Xn[3:self.reservoir.nb * nc:nc])

        z_co2 = Xn[1:self.reservoir.nb * nc:nc] / (1 - z_caco3)
        z_inert = Xn[2:self.reservoir.nb * nc:nc] / (1 - z_caco3)
        z_h2o = Xn[3:self.reservoir.nb * nc:nc] / (1 - z_caco3)

        for ii in range(self.reservoir.nb):
            x_list = Xn[ii*nc:(ii+1)*nc]
            state = value_vector(x_list)
            (sat, x, rho, rho_m, mu, kr, pc, ph) = self.property_container.evaluate(state)

            rel_perm[ii, :] = kr
            visc[ii, :] = mu
            density[ii, :2] = rho
            density_m[ii, :2] = rho_m

            density[2] = self.property_container.solid_dens[-1]

            X[ii, :, 0] = x[1][:-1]
            X[ii, :, 1] = x[0][:-1]
            Sg[ii] = sat[0]
            Ss[ii] = z_caco3[ii]

        # Write all output to a file:
        with open(filename, 'w+') as f:
            # Print headers:
            print('//Gridblock\t gas_sat\t pressure\t C_m\t poro\t co2_liq\t co2_vap\t h2o_liq\t h2o_vap\t ca_plus_co3_liq\t liq_dens\t vap_dens\t solid_dens\t liq_mole_dens\t vap_mole_dens\t solid_mole_dens\t rel_perm_liq\t rel_perm_gas\t visc_liq\t visc_gas', file=f)
            print('//[-]\t [-]\t [bar]\t [kmole/m3]\t [-]\t [-]\t [-]\t [-]\t [-]\t [-]\t [kg/m3]\t [kg/m3]\t [kg/m3]\t [kmole/m3]\t [kmole/m3]\t [kmole/m3]\t [-]\t [-]\t [cP]\t [cP]', file=f)
            for ii in range (self.reservoir.nb):
                print('{:d}\t {:6.5f}\t {:7.5f}\t {:7.5f}\t {:6.5f}\t {:6.5f}\t {:6.5f}\t {:6.5f}\t {:6.5f}\t {:6.5f}\t {:8.5f}\t {:8.5f}\t {:8.5f}\t {:7.5f}\t {:7.5f}\t {:7.5f}\t {:6.5f}\t {:6.5f}\t {:6.5f}\t {:6.5f}'.format(
                    ii, Sg[ii], P[ii], Ss[ii] * density_m[ii, 2], 1 - Ss[ii], X[ii, 0, 0], X[ii, 0, 1], X[ii, 2, 0], X[ii, 2, 1], X[ii, 1, 0],
                    density[ii, 0], density[ii, 1], density[ii, 2], density_m[ii, 0], density_m[ii, 1], density_m[ii, 2],
                    rel_perm[ii, 0], rel_perm[ii, 1], visc[ii, 0], visc[ii, 1]), file=f)

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

        fig, axs = plt.subplots(3, 3, figsize=(12, 10), dpi=200, facecolor='w', edgecolor='k')
        """ sg and x """
        axs[0][0].plot(z_co2, 'b')
        axs[0][0].set_xlabel('x [m]', font_dict_axes)
        axs[0][0].set_ylabel('$z_{CO_2}$ [-]', font_dict_axes)
        axs[0][0].set_title('Fluid composition', fontdict=font_dict_title)

        axs[0][1].plot(z_h2o, 'b')
        axs[0][1].set_xlabel('x [m]', font_dict_axes)
        axs[0][1].set_ylabel('$z_{H_2O}$ [-]', font_dict_axes)
        axs[0][1].set_title('Fluid composition', fontdict=font_dict_title)

        axs[0][2].plot(z_inert, 'b')
        axs[0][2].set_xlabel('x [m]', font_dict_axes)
        axs[0][2].set_ylabel('$z_{w, Ca+2} + z_{w, CO_3-2}$ [-]', font_dict_axes)
        axs[0][2].set_title('Fluid composition', fontdict=font_dict_title)

        axs[1][0].plot(X[:, 0, 0], 'b')
        axs[1][0].set_xlabel('x [m]', font_dict_axes)
        axs[1][0].set_ylabel('$x_{w, CO_2}$ [-]', font_dict_axes)
        axs[1][0].set_title('Liquid mole fraction', fontdict=font_dict_title)

        axs[1][1].plot(X[:, 2, 0], 'b')
        axs[1][1].set_xlabel('x [m]', font_dict_axes)
        axs[1][1].set_ylabel('$x_{w, H_2O}$ [-]', font_dict_axes)
        axs[1][1].set_title('Liquid mole fraction', fontdict=font_dict_title)

        axs[1][2].plot(X[:, 1, 0], 'b')
        axs[1][2].set_xlabel('x [m]', font_dict_axes)
        axs[1][2].set_ylabel('$x_{w, Ca+2} + x_{w, CO_3-2}$ [-]', font_dict_axes)
        axs[1][2].set_title('Liquid mole fraction', fontdict=font_dict_title)

        axs[2][0].plot(P, 'b')
        axs[2][0].set_xlabel('x [m]', font_dict_axes)
        axs[2][0].set_ylabel('$p$ [bar]', font_dict_axes)
        axs[2][0].set_title('Pressure', fontdict=font_dict_title)

        axs[2][1].plot(Sg, 'b')
        axs[2][1].set_xlabel('x [m]', font_dict_axes)
        axs[2][1].set_ylabel('$s_g$ [-]', font_dict_axes)
        axs[2][1].set_title('Gas saturation', fontdict=font_dict_title)

        axs[2][2].plot(1 - Ss, 'b')
        axs[2][2].set_xlabel('x [m]', font_dict_axes)
        axs[2][2].set_ylabel('$\phi$ [-]', font_dict_axes)
        axs[2][2].set_title('Porosity', fontdict=font_dict_title)

        left = 0.05  # the left side of the subplots of the figure
        right = 0.95  # the right side of the subplots of the figure
        bottom = 0.05  # the bottom of the subplots of the figure
        top = 0.95  # the top of the subplots of the figure
        wspace = 0.25  # the amount of width reserved for blank space between subplots
        hspace = 0.25  # the amount of height reserved for white space between subplots
        plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

        for ii in range(3):
            for jj in range(3):
                for tick in axs[ii][jj].xaxis.get_major_ticks():
                    tick.label.set_fontsize(20)

                for tick in axs[ii][jj].yaxis.get_major_ticks():
                    tick.label.set_fontsize(20)

        plt.tight_layout()
        plt.savefig(filename)
        plt.show()

        # Write to numpy array:
        output_data = np.zeros((self.reservoir.nb, 9))
        output_data[:, 0] = z_co2
        output_data[:, 1] = z_h2o
        output_data[:, 2] = z_inert

        output_data[:, 3] = X[:, 0, 0]
        output_data[:, 4] = X[:, 2, 0]
        output_data[:, 5] = X[:, 1, 0]

        output_data[:, 6] = P
        output_data[:, 7] = Sg
        output_data[:, 8] = 1 - Ss
        np.save(filename[:-4], output_data)

    def print_bound_cond(self, type_bc='initial', region='rock', inj_phase='gas'):
        if type_bc == 'initial':
            pressure = self.init_pres  # [bar]
            composition = self.ini_comp[:]
            if region == 'void':
                composition = self.ini_void[:]

            composition += [1 - np.sum(composition)]
            state = np.append(pressure, composition[:-1])
            filename = 'initial_conditions_' + self.physics_type + '_' + region

        elif type_bc == 'injection':
            pressure = self.init_pres  # [bar]
            composition = self.inj_stream_gas[:]
            rate = self.inj_gas_rate
            if inj_phase == 'wat':
                composition = self.inj_stream_wat[:]
                rate = self.inj_wat_rate
            composition += [1 - np.sum(composition)]
            state = np.append(pressure, composition[:-1])
            filename = 'injection_conditions_' + self.physics_type + '_phase_' + inj_phase

        component_list = self.property_container.components_name
        phase_list = ['Vapor', 'Liquid', 'Solid']
        phase_list_dict = ['gas', 'wat', 'sol']
        mole_frac_list = ['Vapor MoleFrac   ', 'Liquid MoleFrac  ', 'Solid MoleFrac   ']
        (sat, x, rho, rho_m, mu, kr, pc, ph) = self.property_container.evaluate(state)
        if len(ph) > 1:
            V = rho[0] * sat[0] / (rho[0] * sat[0] + rho[1] * sat[1])
        else:
            if ph[0] == 0:
                V = 1
            else:
                V = 0

        nc = len(composition)
        zc_sol = composition[-1]
        vap_frac = np.array(x[0])
        liq_frac = np.array(x[1])
        sol_frac = np.array([0] * (nc - 1) + [1])
        phase_frac = np.array([V * (1 - zc_sol), (1 - V) * (1 - zc_sol), zc_sol])
        phase_split_ini = np.array([vap_frac,
                                    liq_frac,
                                    sol_frac])

        with open(filename + '.txt', 'w+') as f:
            # Print phase split, for specific initial conditions:
            if self.combined_ions:
                f.write('Properties based on {:} state: state = [P, z_{:}, z_{:}, z_{:}]\n'.format(type_bc,
                                                                                               component_list[0],
                                                                                               component_list[1],
                                                                                               component_list[2]))
            else:
                f.write('Properties based on {:} state: state = [P, z_{:}, z_{:}, z_{:}, z_{:}]\n'.format(type_bc,
                                                                                               component_list[0],
                                                                                               component_list[1],
                                                                                               component_list[2],
                                                                                               component_list[3]))
            f.write('{:} state = {:}'.format(type_bc, state))

            f.write(
                '\n-----------------------------------------------------------------------------------------------------------------------\n')
            row_header_format = "{:>17}" * (len(component_list) + 1)
            row_format = "{:>17}" + "{:17.7e}" * (len(component_list))

            f.write(row_header_format.format("", *component_list))
            f.write(row_format.format("\nComposition, z_c ", *composition))

            for phase, row in zip(mole_frac_list, phase_split_ini):
                f.write(row_format.format('\n' + phase, *row))
            f.write(
                '\n-----------------------------------------------------------------------------------------------------------------------\n')

            f.write('-------------------------------------------------------------------------------------')
            row_header_format = "{:>27}" * (len(phase_list) + 1)
            f.write(row_header_format.format("\n", *phase_list))

            row_format = "{:>18}" + "{:21.7e}" * (len(phase_list))
            density_list = np.array([])
            viscosity_list = np.array([])

            for phase in phase_list_dict:
                if phase == 'sol':
                    density_list = np.append(density_list, 2000)
                    viscosity_list = np.append(viscosity_list, 0)
                else:
                    density_list = np.append(density_list, self.property_container.density_ev[phase].evaluate(pressure, 0))
                    viscosity_list = np.append(viscosity_list, self.property_container.viscosity_ev[phase].evaluate())

            if V == 1:
                density_molar = np.array([rho_m[0], density_list[1] / 18.015])
            elif V == 0:
                density_molar = np.array([density_list[0] / 44.01, rho_m[1]])
            else:
                density_molar = rho_m
            saturation_list = np.append(
                (phase_frac[:-1] / density_molar) / sum((phase_frac[:-1] / density_molar)) * (1 - zc_sol), zc_sol)
            sat_list_mobile = np.append(saturation_list[:-1] / np.sum(saturation_list[:-1]), 0)
            f.write(row_format.format('\nPhase MoleFrac ', *phase_frac))
            f.write(row_format.format('\nMass Density   ', *density_list))
            f.write(row_format.format('\nViscosity      ', *viscosity_list))
            f.write(row_format.format('\nSat. phi_tot   ', *saturation_list))
            f.write(row_format.format('\nSat. phi_fluid ', *sat_list_mobile))
            f.write('\n-------------------------------------------------------------------------------------')
            f.write('\n')
        return 0

    def print_and_plot_2D(self):
        if self.combined_ions:
            plot_labels = ['$z_{w, Ca+2} + z_{w, CO_3-2}$ [-]', '$x_{w, Ca+2} + x_{w, CO_3-2}$ [-]']
        else:
            plot_labels = ['$z_{w, Ca+2}$ [-]', '$x_{w, Ca+2}$ [-]']

        import matplotlib.pyplot as plt
        font_dict_title = {'family': 'sans-serif',
                           'color': 'black',
                           'weight': 'normal',
                           'size': 16,
                           }

        font_dict_axes = {'family': 'monospace',
                          'color': 'black',
                          'weight': 'normal',
                          'size': 14,
                          }

        nc = self.property_container.nc
        Sg = np.zeros(self.reservoir.nb)
        Ss = np.zeros(self.reservoir.nb)
        X = np.zeros((self.reservoir.nb, nc - 1, 2))
        Xn = np.array(self.physics.engine.X, copy=True)
        z_caco3 = 1 - (
                    Xn[1:self.reservoir.nb * nc:nc] + Xn[2:self.reservoir.nb * nc:nc] + Xn[3:self.reservoir.nb * nc:nc])

        z_co2 = Xn[1:self.reservoir.nb * nc:nc] / (1 - z_caco3)
        z_inert = Xn[2:self.reservoir.nb * nc:nc] / (1 - z_caco3)
        z_h2o = Xn[3:self.reservoir.nb * nc:nc] / (1 - z_caco3)

        for ii in range(self.reservoir.nb):
            x_list = Xn[ii * nc:(ii + 1) * nc]
            state = value_vector(x_list)
            (sat, x, rho, rho_m, mu, kr, pc, ph) = self.property_container.evaluate(state)

            X[ii, :, 0] = x[1][:-1]
            X[ii, :, 1] = x[0][:-1]
            Sg[ii] = sat[0]
            Ss[ii] = z_caco3[ii]

        fig, axs = plt.subplots(4, 2, figsize=(12, 10), dpi=300, facecolor='w', edgecolor='k')
        """ sg and x """
        im0 = axs[0][0].imshow(z_co2.reshape(self.ny, self.nx))
        axs[0][0].set_title('$z_{CO_2}$ [-]', fontdict=font_dict_title)
        plt.colorbar(im0, ax=axs[0][0])

        im1 = axs[0][1].imshow(z_h2o.reshape(self.ny, self.nx))
        axs[0][1].set_title('$z_{H_2O}$ [-]', fontdict=font_dict_title)
        plt.colorbar(im1, ax=axs[0][1])

        im2 = axs[1][0].imshow(z_inert.reshape(self.ny, self.nx))
        axs[1][0].set_title('$z_{w, Ca+2} + z_{w, CO_3-2}$ [-]', fontdict=font_dict_title)
        plt.colorbar(im2, ax=axs[1][0])

        im3 = axs[1][1].imshow(X[:, 0, 0].reshape(self.ny, self.nx))
        axs[1][1].set_title('$x_{w, CO_2}$ [-]', fontdict=font_dict_title)
        plt.colorbar(im3, ax=axs[1][1])

        im4 = axs[2][0].imshow(X[:, 2, 0].reshape(self.ny, self.nx))
        axs[2][0].set_title('$x_{w, H_2O}$ [-]', fontdict=font_dict_title)
        plt.colorbar(im4, ax=axs[2][0])

        im5 = axs[2][1].imshow(X[:, 1, 0].reshape(self.ny, self.nx))
        axs[2][1].set_title('$x_{w, Ca+2} + x_{w, CO_3-2}$ [-]', fontdict=font_dict_title)
        plt.colorbar(im5, ax=axs[2][1])

        im6 = axs[3][0].imshow(Sg.reshape(self.ny, self.nx))
        axs[3][0].set_title('$s_g$ [-]', fontdict=font_dict_title)
        plt.colorbar(im6, ax=axs[3][0])

        im7 = axs[3][1].imshow(1 - z_caco3.reshape(self.ny, self.nx))
        axs[3][1].set_title('$\phi$ [-]', fontdict=font_dict_title)
        plt.colorbar(im7, ax=axs[3][1])

        left = 0.05  # the left side of the subplots of the figure
        right = 0.95  # the right side of the subplots of the figure
        bottom = 0.05  # the bottom of the subplots of the figure
        top = 0.95  # the top of the subplots of the figure
        wspace = 0.25  # the amount of width reserved for blank space between subplots
        hspace = 0.25  # the amount of height reserved for white space between subplots
        plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

        for ii in range(4):
            for jj in range(2):
                for tick in axs[ii][jj].xaxis.get_major_ticks():
                    tick.label.set_fontsize(14)

                for tick in axs[ii][jj].yaxis.get_major_ticks():
                    tick.label.set_fontsize(14)

        plt.tight_layout()
        plt.savefig("results_kinetic_2D.pdf")
        plt.show()

        fig1 = plt.subplots(2, 1)
        prof = 8
        plt.subplot(211)
        plt.plot(z_co2.reshape(self.ny, self.nx)[:, prof], 'b')
        plt.plot(z_inert.reshape(self.ny, self.nx)[:, prof], 'r')
        plt.grid()
        plt.legend(['$z_{CO2}$', '$z_{ions}$'])
        plt.subplot(212)
        plt.plot(Sg.reshape(self.ny, self.nx)[:, prof], 'b')
        plt.plot(1 - z_caco3.reshape(self.ny, self.nx)[:, prof], 'r')
        plt.grid()
        plt.legend(['$s_g$', '$\phi$'])
        plt.tight_layout()
        plt.savefig("results_kinetic_2D_slices_y={:}.pdf".format(prof))
        plt.show()

        plt.subplot(211)
        plt.plot(z_co2.reshape(self.ny, self.nx)[prof, :], 'b')
        plt.plot(z_inert.reshape(self.ny, self.nx)[prof, :], 'r')
        plt.grid()
        plt.legend(['$z_{CO2}$', '$z_{ions}$'])
        plt.subplot(212)
        plt.plot(Sg.reshape(self.ny, self.nx)[prof, :], 'b')
        plt.plot(1 - z_caco3.reshape(self.ny, self.nx)[prof, :], 'r')
        plt.grid()
        plt.legend(['$s_g$', '$\phi$'])
        plt.tight_layout()
        plt.savefig("results_kinetic_2D_slices_x={:}.pdf".format(prof))
        plt.show()
        return 0


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

        for i in range(self.nc - 1):
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
