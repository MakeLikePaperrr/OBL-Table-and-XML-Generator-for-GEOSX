import numpy as np
import os
import sys
import progressbar
from time import sleep


class OBLTableGenerator:
    """
        Using Delft Advanced Research Terra Simulator (DARTS) to generate operators for any OBL implementation using
        static interpolation.
    """
    def __init__(self, model, physical_space_constraint=False):
        """
        Class constructor
        :param model: Fully instantiated specific DARTS model
        :param physical_space_constraint: if True generate a reduced table (saves a lot of space) containing only the
                                          physical states in the parameters and slightly across this boundary
        """
        self.model = model
        self.ne = model.physics.nc
        self.thermal = model.thermal
        self.n_vars = model.physics.n_vars
        self.n_ops = model.physics.n_ops
        self.obl_min_axis = np.array(model.physics.n_axes_min)
        self.obl_max_axis = np.array(model.physics.n_axes_max)
        self.obl_num_pts = np.array(model.physics.n_axes_points)
        self.obl_step_size = (self.obl_max_axis - self.obl_min_axis) / (self.obl_num_pts - 1)
        self.cum_prod_obl_pts = np.zeros((self.n_vars,), dtype=np.longlong)
        self.cum_prod_obl_pts[0] = self.obl_num_pts[0]
        for i in range(1, self.n_vars):
            self.cum_prod_obl_pts[i] = self.cum_prod_obl_pts[i - 1] * self.obl_num_pts[i]
        self.tot_op_vals = self.cum_prod_obl_pts[-1]
        self.operator_table = np.zeros((self.tot_op_vals, self.n_ops), dtype=np.single)
        self.all_states = np.zeros((self.tot_op_vals, self.n_vars), dtype=np.single)

        for i in range(self.n_vars):
            param_vec = np.linspace(self.obl_min_axis[i], self.obl_max_axis[i], num=self.obl_num_pts[i], endpoint=True)
            # self.all_states[:, i] = np.tile(np.repeat(param_vec, int(self.tot_op_vals / (self.cum_prod_obl_pts[i]))),
            #                                 int(self.cum_prod_obl_pts[i] / self.cum_prod_obl_pts[0]))
            self.all_states[:, i] = np.tile(np.repeat(param_vec, int(self.tot_op_vals / (self.cum_prod_obl_pts[i]))),
                                            int(self.tot_op_vals / (int(self.tot_op_vals / (self.cum_prod_obl_pts[i])) * len(param_vec))))
        # Determine physical states:
        #   Generate operator values for each state and don't calculate for unphysical values far
        #   far away from the physical boundary! (only for last component!!!)
        #   formula: 0 - (nc - 1) * Delta omega < z_nc < 1 + (nc - 1) * Delta omega is the range of z_nc since otherwise
        #   for sure too far from physical region
        self.physical_ids = np.arange(0, self.tot_op_vals, dtype=np.longlong)
        if physical_space_constraint:
            if not (self.thermal == 1 and self.n_vars == 2):
                z_nc = 1 - np.sum(self.all_states[:, 1:self.ne], axis=1)
                if self.thermal == 1:
                    d_zc = self.obl_step_size[-2]
                else:
                    d_zc = self.obl_step_size[-1]
                self.physical_ids = np.where((z_nc >= (-(self.ne - 1) * d_zc + 0)) *
                                             (z_nc <= (1 + (self.ne - 1) * d_zc)))[0]

    def generate_table(self):
        """
        Loop over all physical states in parameter space to generate OBL tabel using specific DARTS model
        :return:
        """
        tot_iters = len(self.physical_ids)
        step_size = round(tot_iters / 100)
        steps = np.arange(step_size, tot_iters, step_size)
        count = 0
        for i in self.physical_ids:
            temp_state = np.array(self.all_states[i, :], copy=True)
            temp_operators = np.zeros((self.n_ops,))
            self.model.physics.acc_flux_etor.evaluate(temp_state, temp_operators)
            self.operator_table[i, :] = temp_operators
            count += 1
            if any(steps == count):
                # Print iterations progress (solution from: https://stackoverflow.com/a/3002100, last accessed 27-09-2022)
                sys.stdout.write('\r')
                sys.stdout.write(f'Computing operator values: '
                                 f'[{"=" * round(count / tot_iters * 100) + "-" * round((1 - count / tot_iters) * 100) }]'
                                 f' {round(count / tot_iters * 100)}%')
                sys.stdout.flush()
        sys.stdout.write('\r')
        sys.stdout.write(f'Computing operator values: [{"=" * 100}] {100}%')
        sys.stdout.flush()
        sys.stdout.write(f'\nSuccess :)!')
        return 0

    def write_table_to_file(self, filename):
        """
        After generating tables based on specific DARTS model, this method can generate tables for GEOSX to read
        :param filename: name of the file
        :return:
        """
        sys.stdout.write(f'\nStart writing table to file (can take a while for large num_comp!)')
        with open(filename, 'w+') as f:
            # Write parameters of linearization: number of nonlinear unknowns, number of operators per state
            f.write('{:} {:}\n'.format(self.n_vars, self.n_ops))
            for i in range(self.n_vars):
                # Write the bounds for each parameter dimenson: num points, min parameter, max parameter
                f.write('{:} {:} {:}\n'.format(self.obl_num_pts[i], self.obl_min_axis[i], self.obl_max_axis[i]))
            # Write for each state, ordered based on ... the value of the operators
            for i in range(self.tot_op_vals):
                # f.write(self.operator_table[:, i])
                f.write(" ".join(map(str, self.operator_table[i, :])) + "\n")
        sys.stdout.write(f'\nSuccess :)!')
        return 0
