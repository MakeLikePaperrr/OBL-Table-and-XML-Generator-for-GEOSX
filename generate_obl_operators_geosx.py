from geosx_obl_operators_object import OBLTableGenerator
import numpy as np
import sys
sys.path.append('physics_sup')
from benchmark_model.model import Model as ModelBenchmark
from simple_kinetics.model import Model as ModelKinBasic
from binary_system.model import Model as ModelBinarySystem


case = 3
if case == 1:
    my_model = ModelBinarySystem()
    file_name = 'binary_system_operators.txt'
    physical_constraint = False
elif case == 2:
    my_model = ModelKinBasic()
    file_name = 'simple_kin_operators.txt'
    physical_constraint = True
elif case == 3:
    file_name = 'benchmark_operators.txt'
    my_model = ModelBenchmark(obl_pts=[32, 42, 42, 42], min_z_vec=[1e-13, 1e-13, 1e-13], max_z_vec=[1-1e-13, 0.5, 1-1e-13])
    physical_constraint = True

my_obl_table_gen = OBLTableGenerator(my_model, physical_space_constraint=physical_constraint)
my_obl_table_gen.generate_table()
my_obl_table_gen.write_table_to_file(filename=file_name)
