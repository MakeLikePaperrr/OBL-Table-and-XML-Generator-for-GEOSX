import sys
sys.path.append('physics_sup/')
import meshio
import matplotlib.pyplot as plt
from benchmark_model.model import Model as ModelBenchmark
from simple_kinetics.model import Model as ModelSimpleKin
from binary_system.model import Model as ModelBinarySystem
import numpy as np


case = 3
if case == 1:
    plot_size_x = 12
    plot_size_y = 10
    n = ModelBinarySystem()
    DIR = 'binary_system_final_dT.vtu'
    filename = "results_binary_system_geosx.pdf"
    plot_titles = [['Fluid composition', 'Fluid composition'],
                   ['Pressure', 'Gas saturation']]
    axis_names = [['$z_{CO_2}$ [-]', '$z_{H_2O}$ [-]'],
                  ['$p$ [bar]', '$s_g$ [-]']]
    file_name_darts_output = 'results_binary_system_darts.npy'
elif case == 2:
    plot_size_x = 16
    plot_size_y = 7
    n = ModelSimpleKin()
    DIR = 'simple_kinetics_final_dT.vtu'
    filename = "results_simple_kinetics_geosx.pdf"
    plot_titles = [['Fluid composition', 'Fluid composition', 'Fluid composition'],
                   ['empty', 'Pressure', 'Porosity']]
    axis_names = [['$z_{H_2O}$ [-]', '$z_{ions}$ [-]', '$z_{CaCO_3}$ [-]'],
                  ['empty', '$p$ [bar]', '$\phi$ [-]']]
    file_name_darts_output = 'results_simple_kinetics_darts.npy'
elif case == 3:
    plot_size_x = 12
    plot_size_y = 10
    n = ModelBenchmark(grid_1D=True)
    DIR = 'chemical_benchmark_final_dT.vtu'
    filename = "results_chemical_benchmark_geosx.pdf"
    plot_titles = [['Fluid composition', 'Fluid composition', 'Fluid composition'],
                   ['Liquid mole fraction', 'Liquid mole fraction', 'Liquid mole fraction'],
                   ['Pressure', 'Gas saturation', 'Porosity']]
    axis_names = [['$z_{CO_2}$ [-]', '$z_{H_2O}$ [-]', '$z_{ions}$ [-]'],
                  ['$x_{w,CO_2}$ [-]', '$x_{w,H_2O}$ [-]', '$x_{w,ions}$ [-]'],
                  ['$p$ [bar]', '$s_g$ [-]', '$\phi$ [-]']]
    file_name_darts_output = f'results_chemical_benchmark_darts_{512}.npy'

index_start = 0
index_stop = n.reservoir.nb - 0
mesh = meshio.read(DIR)
pres = mesh.cell_data_dict['pressure']['hexahedron'][index_start:index_stop] / 1e5
s_g = np.zeros((index_stop - index_start,))

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

def read_darts_output(filename, nx, ny, id_start, id_stop):
    data_array = np.load(filename)
    if ny == 2:
        data = [[], []]
    else:
        data = [[], [], []]
    count = 0
    for j in range(ny):
        for i in range(nx):
            data[j].append(data_array[id_start:id_stop, count])
            count += 1
    return data

if case == 1:
    z_co2 = mesh.cell_data_dict['globalCompFraction']['hexahedron'][index_start:index_stop, 0]
    for i in range(index_stop - index_start):
        state = np.array([pres[i], z_co2[i]])
        (sat, x, rho, rho_m, mu, kr, pc, ph) = n.property_container.evaluate(state)
        s_g[i] = sat[0]

    data_arrays = [[z_co2, 1 - z_co2],
                   [pres, s_g]]
elif case == 2:
    z_h2o = mesh.cell_data_dict['globalCompFraction']['hexahedron'][index_start:index_stop, 0]
    z_ions = mesh.cell_data_dict['globalCompFraction']['hexahedron'][index_start:index_stop, 1]
    for i in range(index_stop - index_start):
        state = np.array([pres[i], z_h2o[i], z_ions[i]])
        (sat, x, rho, rho_m, mu, kr, pc, ph) = n.property_container.evaluate(state)
        s_g[i] = sat[0]

    z_caco3 = 1 - z_h2o - z_ions
    zc_fl_h2o = z_h2o / (1 - z_caco3)
    zc_fl_ions = z_ions / (1 - z_caco3)
    data_arrays = [[zc_fl_h2o, zc_fl_ions, z_caco3],
                   ['empty', pres, 1 - z_caco3]]
elif case == 3:
    z_co2 = mesh.cell_data_dict['globalCompFraction']['hexahedron'][index_start:index_stop, 0]
    z_h2o = mesh.cell_data_dict['globalCompFraction']['hexahedron'][index_start:index_stop, 2]
    z_ions = mesh.cell_data_dict['globalCompFraction']['hexahedron'][index_start:index_stop, 1]
    z_caco3 = 1 - z_co2 - z_h2o - z_ions
    zc_fl_co2 = z_co2 / (1 - z_caco3)
    zc_fl_h2o = z_h2o / (1 - z_caco3)
    zc_fl_ions = z_ions / (1 - z_caco3)
    X = np.zeros((n.reservoir.nb, 3))
    for i in range(index_stop - index_start):
        state = np.array([pres[i], z_co2[i], z_ions[i], z_h2o[i]])
        (sat, x, rho, rho_m, mu, kr, pc, ph) = n.property_container.evaluate(state)
        s_g[i] = sat[0]
        X[i, :] = x[1][:-1]

    data_arrays = [[zc_fl_co2, zc_fl_h2o, zc_fl_ions],
                   [X[:, 0], X[:, 2], X[:, 1]],
                   [pres, s_g, 1 - z_caco3]]

plot_x = len(plot_titles[0])
plot_y = len(plot_titles)
darts_data_array = read_darts_output(file_name_darts_output, plot_x, plot_y, index_start, index_stop)

fig, axs = plt.subplots(plot_y, plot_x, figsize=(plot_size_x, plot_size_y), dpi=400, facecolor='w', edgecolor='k')
for i in range(plot_x):
    for j in range(plot_y):
        if plot_titles[j][i] == 'empty':
            continue

        axs[j][i].plot(darts_data_array[j][i], 'o',
                       markerfacecolor="None", markeredgecolor='red', markeredgewidth=1, label='DARTS')
        axs[j][i].plot(data_arrays[j][i], 'b', linewidth=2, label='GEOSX')
        axs[j][i].set_xlabel('x [m]', font_dict_axes)
        axs[j][i].set_ylabel(axis_names[j][i], font_dict_axes)
        axs[j][i].set_title(plot_titles[j][i], fontdict=font_dict_title)

axs[0][0].legend()
left = 0.05  # the left side of the subplots of the figure
right = 0.95  # the right side of the subplots of the figure
bottom = 0.05  # the bottom of the subplots of the figure
top = 0.95  # the top of the subplots of the figure
wspace = 0.25  # the amount of width reserved for blank space between subplots
hspace = 0.25  # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

for i in range(plot_x):
    for j in range(plot_y):
        for tick in axs[j][i].xaxis.get_major_ticks():
            tick.label.set_fontsize(20)

        for tick in axs[j][i].yaxis.get_major_ticks():
            tick.label.set_fontsize(20)

plt.tight_layout()
plt.savefig(filename)
plt.show()


with open('GEOSX_11_t1000.csv', 'w+') as f:
    # Print headers:
    print('x, S_g, P_g, phi, x_H2O, x_CO2, x_Ca+CO3', file=f)
    for ii in range(n.reservoir.nb):
        print(f'{ii * 1 + 0.5}, {data_arrays[2][1][ii]}, {data_arrays[2][0][ii] * 1e5}, {data_arrays[2][2][ii]}, '
              f'{data_arrays[1][1][ii]}, {data_arrays[1][0][ii]}, {data_arrays[1][2][ii]}',
              file=f)
