import numpy as np
import os


DIR = 'perm_fields_geosx'
mu_perm = 100
Lx = 200
Ly = 800
nx = 20
ny = 80
nb = nx * ny
solid_init = 0.7
trans_exp = 3
perm_coarse = (mu_perm + (np.random.rand(nb) * 2 - 1) * 0.05 * mu_perm) / ((1 - solid_init) ** trans_exp)

dx = Lx / nx
dy = Ly / ny

centroids_x = np.linspace(dx / 2, nx * dx - dx / 2, nx)
centroids_y = np.linspace(dy / 2, ny * dy - dy / 2, ny)
centroids_z = np.array([5.0])

np.savetxt(os.path.join(DIR, 'xlin.geos'), centroids_x)
np.savetxt(os.path.join(DIR, 'ylin.geos'), centroids_y)
np.savetxt(os.path.join(DIR, 'zlin.geos'), centroids_z)

np.savetxt(os.path.join(DIR, 'permx.geos'), perm_coarse)
np.savetxt(os.path.join(DIR, 'permy.geos'), perm_coarse)
np.savetxt(os.path.join(DIR, 'permz.geos'), perm_coarse)
