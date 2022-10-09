from geosx_xml_object import SolversData, MeshData, GeometryData, EventsData, ConstitutiveData, \
    InitialConditionData, BoundaryConditionData, RockParamData, FunctionData, PermPoroFunctionData, \
    XMLFileGEOSXGenerator, ElementRegionData, SourceTermData


# Generate XML file:
comp_name = ['Comp_H2O', 'Comp_Ions', 'Comp_CaCO3']
solvers_data = SolversData(num_comp=len(comp_name), num_phases=2)

mesh_data = MeshData(x_coords=(0, 200), y_coords=(0, 800), z_coords=(0, 10),
                     nx=(40), ny=(160), nz=(1), cell_block_name=['matrix00'])

# Define geometries for boundary conditions:
geometry_data = list()
geometry_data.append(GeometryData(x_min=(-0.01, -0.01, -0.01),
                                  x_max=(5.01, 800.01, 10.01),
                                  name='source'))
geometry_data.append(GeometryData(x_min=(194.99, -0.01, -0.01),
                                  x_max=(200.01, 800.01, 10.01),
                                  name='sink'))

# Prescribe simulation and output related parameters:
end_first_tz = 1e4
end_final = 5e7
events_data = EventsData(max_time=end_final, out_name='outputs', out_freq=1e6, out_tar_dir='/Outputs/vtkOutput',
                         solver_names=('solver_1', 'solver_2'), solver_dts=(1e3, 5e5), solver_end_time=(end_first_tz, end_final),
                         solver_begin_time=(0, end_first_tz))

constitutive_data = ConstitutiveData(perm=(3.7e-12, 3.7e-12, 3.7e-12))

# Define regions, should be consistent with mesh_data regions:
elements_regions_data = list()
elements_regions_data.append(ElementRegionData(name='matrix',
                                               region_name=['matrix00']))

# Define parameters for initial and boundary conditions (for each region!):
initial_condition_data = list()
initial_condition_data.append(InitialConditionData(pres_val=1e7, temp_val=348.15, comp_val=[0.1499999, 0.1499999, 0.7],
                                comp_name=comp_name, region_name='matrix'))

boundary_condition_data = list()
# boundary_condition_data.append(BoundaryConditionData(pres_val=1.25e7, temp_val=348.15,
#                                                      comp_val=[0.999999, 1e-10, 1e-10],
#                                                      comp_name=comp_name, region_name='matrix', source_name='source'))
boundary_condition_data.append(BoundaryConditionData(pres_val=7.5e6, temp_val=348.15,
                                                     comp_val=[0.1499999, 0.1499999, 0.7],
                                                     comp_name=comp_name, region_name='matrix', source_name='sink'))

source_term_data = list()
source_term_data.append(SourceTermData(name='sourceTerm', region_name='matrix',
                                       component=0, scale=-1.5,
                                       source_name='source'))

# Define all rock related parameters (again, per region!):
rock_params_name = ['rockHeatCap', 'rockThermalConductivity', 'rockKineticRateFactor']
rock_params_fieldname = ['rockVolumetricHeatCapacity', 'rockThermalConductivity', 'rockKineticRateFactor']
rock_params_value = [2200, 181.44, 1.0]
rock_parameter_data = list()
rock_parameter_data.append(RockParamData(rock_params_name=rock_params_name, rock_params_fieldname=rock_params_fieldname,
                                         rock_params_value=rock_params_value, region_name='matrix'))

# Define tables for permeability functions:
table_function_data = list()
table_function_data.append(FunctionData(name='permxFunc', filename='permx.geos', interpolation='nearest'))
table_function_data.append(FunctionData(name='permyFunc', filename='permy.geos', interpolation='nearest'))
table_function_data.append(FunctionData(name='permzFunc', filename='permz.geos', interpolation='nearest'))

perm_poro_functions = list()
perm_poro_functions.append(PermPoroFunctionData(name='permx', component=0, fun_name='permxFunc',
                                                scale=1e-15, region_name='matrix'))
perm_poro_functions.append(PermPoroFunctionData(name='permy', component=1, fun_name='permyFunc',
                                                scale=1e-15, region_name='matrix'))
perm_poro_functions.append(PermPoroFunctionData(name='permz', component=2, fun_name='permzFunc',
                                                scale=1e-15, region_name='matrix'))

# Construct XML file and write to file:
my_xml_file = XMLFileGEOSXGenerator(file_name='simple_kinetics_with_tables.xml', solvers_data=solvers_data, mesh_data=mesh_data,
                                    geometry_data=geometry_data, events_data=events_data,
                                    constitutive_data=constitutive_data, elements_regions_data=elements_regions_data,
                                    boundary_condition_data=boundary_condition_data,
                                    table_function_data=table_function_data, rock_parameter_data=rock_parameter_data,
                                    initial_condition_data=initial_condition_data,
                                    perm_poro_functions=perm_poro_functions, source_term_data=source_term_data,
                                    obl_table_name='simple_kin_operators.txt', region_name=['matrix'])
my_xml_file.write_to_file()
