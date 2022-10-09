from geosx_xml_object import SolversData, MeshData, GeometryData, EventsData, ConstitutiveData, \
    InitialConditionData, BoundaryConditionData, RockParamData, FunctionData, PermPoroFunctionData, \
    XMLFileGEOSXGenerator, ElementRegionData, SourceTermData


# Generate XML file:
comp_name = ['Comp_CO2', 'Comp_Ions', 'Comp_H2O', 'Comp_CaCO3']
solvers_data = SolversData(num_comp=len(comp_name), num_phases=2)

mesh_data = MeshData(x_coords=(0, 120, 480, 600), y_coords=(0, 10), z_coords=(0, 80, 160, 240),
                     nx=(24, 72, 24), ny=(1), nz=(16, 16, 16),
                     cell_block_name=['matrix00', 'matrix01', 'matrix02',
                                      'matrix10', 'gap11', 'matrix12',
                                      'matrix20', 'matrix21', 'matrix22'])

# Define geometries for boundary conditions:
geometry_data = list()
geometry_data.append(GeometryData(x_min=(-0.01, -0.01, -0.01),
                                  x_max=(5.01, 10.01, 120.01),
                                  name='source_gas'))
geometry_data.append(GeometryData(x_min=(-0.01, -0.01, 119.99),
                                  x_max=(5.01, 10.01, 240.01),
                                  name='source_water'))
geometry_data.append(GeometryData(x_min=(594.99, -0.01, -0.01),
                                  x_max=(600.01, 10.01, 240.01),
                                  name='sink'))

# Prescribe simulation and output related parameters:
end_first_tz = 1e4
end_final = 8.64e7
events_data = EventsData(max_time=end_final, out_name='outputs', out_freq=1e6, out_tar_dir='/Outputs/vtkOutput',
                         solver_names=('solver_1', 'solver_2'), solver_dts=(1e3, 1e4), solver_end_time=(end_first_tz, end_final),
                         solver_begin_time=(0, end_first_tz))

constitutive_data = ConstitutiveData(perm=(3.7e-12, 3.7e-12, 3.7e-12))

# Define regions, should be consistent with mesh_data regions:
elements_regions_data = list()
elements_regions_data.append(ElementRegionData(name='matrix',
                                               region_name=['matrix00', 'matrix01', 'matrix02',
                                                            'matrix10',             'matrix12',
                                                            'matrix20', 'matrix21', 'matrix22']))
elements_regions_data.append(ElementRegionData(name='gap',
                                               region_name=['gap11']))

# Define parameters for initial and boundary conditions (for each region!):
initial_condition_data = list()
initial_condition_data.append(InitialConditionData(pres_val=1e7, temp_val=348.15,
                                                   comp_val=[1e-10, 0.1499999, 0.1499999, 0.7],
                                                   comp_name=comp_name, region_name='matrix'))
initial_condition_data.append(InitialConditionData(pres_val=1e7, temp_val=348.15,
                                                   comp_val=[1e-10, 0.399999, 0.399999, 0.2],
                                                   comp_name=comp_name, region_name='gap'))

boundary_condition_data = list()
boundary_condition_data.append(BoundaryConditionData(pres_val=9.5e6, temp_val=348.15,
                                                     comp_val=[1e-10, 0.1499999, 0.1499999, 0.7],
                                                     comp_name=comp_name, region_name='matrix', source_name='sink'))

source_term_data = list()
kmol_gas_per_sec = (100 / 44.01) * 1000 / (60 * 60 * 24)
kmol_water_per_sec = (1000 / 18.02) * 200 / (60 * 60 * 24)
source_term_data.append(SourceTermData(name='sourceGas', region_name='matrix',
                                       component=0, scale=-kmol_gas_per_sec,
                                       source_name='source_gas'))
source_term_data.append(SourceTermData(name='sourceWater', region_name='matrix',
                                       component=2, scale=-kmol_water_per_sec,
                                       source_name='source_water'))

# Define all rock related parameters (again, per region!):
rock_params_name = ['rockHeatCap', 'rockThermalConductivity', 'rockKineticRateFactor']
rock_params_fieldname = ['rockVolumetricHeatCapacity', 'rockThermalConductivity', 'rockKineticRateFactor']
rock_params_value = [2200, 181.44, 1.0]
rock_parameter_data = list()
rock_parameter_data.append(RockParamData(rock_params_name=rock_params_name, rock_params_fieldname=rock_params_fieldname,
                                         rock_params_value=rock_params_value, region_name='matrix'))
rock_parameter_data.append(RockParamData(rock_params_name=rock_params_name, rock_params_fieldname=rock_params_fieldname,
                                         rock_params_value=rock_params_value, region_name='gap'))

# Construct XML file and write to file:
my_xml_file = XMLFileGEOSXGenerator(file_name='benchmark_2D_with_source.xml', solvers_data=solvers_data, mesh_data=mesh_data,
                                    geometry_data=geometry_data, events_data=events_data,
                                    constitutive_data=constitutive_data, elements_regions_data=elements_regions_data,
                                    boundary_condition_data=boundary_condition_data,
                                    table_function_data=None, rock_parameter_data=rock_parameter_data,
                                    initial_condition_data=initial_condition_data,
                                    perm_poro_functions=None, source_term_data=source_term_data,
                                    obl_table_name='benchmark_operators.txt', region_name=['matrix', 'gap'])
my_xml_file.write_to_file()
