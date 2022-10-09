import sys
import os
import numpy as np


class XMLFileGEOSXGenerator:
    """
        Automatically generate basic XML file using some pre-defined or user specified input parameters, mainly to be
        used in conjection with geosx_operator_generation for using OBL in GEOSX
    """
    def __init__(self, xml_version=1.0, file_name='my_xml_file.xml', obl_table_name='simple_kin_operators.txt',
                 region_name=None, flow_name='compflow', flow_discretization='fluidTPFA',
                 solvers_data=None, mesh_data=None, geometry_data=None, events_data=None,
                 numerical_methods_data=None, elements_regions_data=None,
                 constitutive_data=None, boundary_condition_data=None, outputs_data=None,
                 table_function_data=None, rock_parameter_data=None,
                 initial_condition_data=None, perm_poro_functions=None,
                 source_term_data=None):
        self.file_name = file_name
        self.obl_table_name = obl_table_name
        self.flow_discretization = flow_discretization
        self.cell_block_name = mesh_data.cell_block_name
        self.base_template = lambda model: f'<?xml version="{xml_version}"?>\n<Problem>\n{model}</Problem>\n'
        self.flow_name = flow_name
        self.full_xml_file = f''
        self.full_model = f''

        self.solvers_data = solvers_data
        self.mesh_data = mesh_data
        self.geometry_data = geometry_data
        self.events_data = events_data
        self.numerical_methods_data = numerical_methods_data
        self.element_regions_data = elements_regions_data
        self.constitutive_data = constitutive_data
        self.boundary_condition_data = boundary_condition_data
        self.rock_parameter_data = rock_parameter_data
        self.initial_condition_data = initial_condition_data

        if perm_poro_functions is None:
            self.perm_poro_functions = []
        else:
            self.perm_poro_functions = perm_poro_functions
        if table_function_data is None:
            self.table_function_data = []
        else:
            self.table_function_data = table_function_data
        self.outputs_data = outputs_data
        if region_name is None:
            self.region_name = ['Region1']
        else:
            self.region_name = region_name
        if source_term_data is None:
            self.source_term_data = []
        else:
            self.source_term_data = source_term_data

        if self.solvers_data is not None:
            self.add_solvers()
        if self.mesh_data is not None:
            self.add_mesh()
        if self.geometry_data is not None:
            self.add_geometry()
        if self.events_data is not None:
            self.add_events()
        self.add_numerical_methods()
        self.add_elements_regions()
        if self.constitutive_data is not None:
            self.add_constitutive()
        self.add_field_specifications()
        if self.table_function_data:
            self.add_table_functions()
        self.add_outputs()

        self.full_xml_file = self.base_template(self.full_model)

    def add_solvers(self):
        self.full_model += f'\t<Solvers>\n' \
                           f'\t\t<ReactiveCompositionalMultiphaseOBL\n' \
                           f'\t\t\tname="{self.flow_name}"\n' \
                           f'\t\t\tlogLevel="{self.solvers_data.log_level}"\n' \
                           f'\t\t\tdiscretization="{self.flow_discretization}"\n' \
                           f'\t\t\ttargetRegions="{{{", ".join([str(ii) for ii in self.region_name])}}}"\n' \
                           f'\t\t\tenableEnergyBalance="{self.solvers_data.energy_balance}"\n' \
                           f'\t\t\tmaxCompFractionChange="{self.solvers_data.max_comp_frac_change}"\n' \
                           f'\t\t\tnumComponents="{self.solvers_data.num_comp}"\n' \
                           f'\t\t\tnumPhases="{self.solvers_data.num_phases}"\n' \
                           f'\t\t\ttransMultExp="{self.solvers_data.trans_mult_exp}"\n' \
                           f'\t\t\tOBLOperatorsTableFile="{self.obl_table_name}">\n' \
                           f'\t\t<NonlinearSolverParameters\n' \
                           f'\t\t\ttimeStepDecreaseFactor="{self.solvers_data.time_step_cut_fac}"\n' \
                           f'\t\t\tnewtonTol="{self.solvers_data.newton_tolerance}"\n' \
                           f'\t\t\tnewtonMaxIter="{self.solvers_data.newton_max_iters}"/>\n'\
                           f'\t\t<LinearSolverParameters\n' \
                           f'\t\t\tdirectParallel="{self.solvers_data.direct_parallel}"/>\n' \
                           f'\t\t</ReactiveCompositionalMultiphaseOBL>\n' \
                           f'\t</Solvers>\n\n'

    def add_mesh(self):
        self.full_model += f'\t<Mesh>\n' \
                           f'\t\t<InternalMesh\n' \
                           f'\t\t\tname="{self.mesh_data.name_mesh}"\n' \
                           f'\t\t\telementTypes="{{{self.mesh_data.element_types}}}"\n' \
                           f'\t\t\txCoords="{{{", ".join([str(ii) for ii in self.mesh_data.x_coords])}}}"\n' \
                           f'\t\t\tyCoords="{{{", ".join([str(ii) for ii in self.mesh_data.y_coords])}}}"\n' \
                           f'\t\t\tzCoords="{{{", ".join([str(ii) for ii in self.mesh_data.z_coords])}}}"\n' \
                           f'\t\t\tnx="{{{", ".join([str(ii) for ii in self.mesh_data.nx])}}}"\n' \
                           f'\t\t\tny="{{{", ".join([str(ii) for ii in self.mesh_data.ny])}}}"\n' \
                           f'\t\t\tnz="{{{", ".join([str(ii) for ii in self.mesh_data.nz])}}}"\n' \
                           f'\t\t\tcellBlockNames="{{{", ".join(self.mesh_data.cell_block_name)}}}"/>\n' \
                           f'\t</Mesh>\n\n'

    def add_geometry(self):
        box_all = f''
        for ith_box in self.geometry_data:
            box_all += f'\t\t<Box\n' \
                       f'\t\t\tname="{ith_box.name}"\n' \
                       f'\t\t\txMin="{{{ith_box.x_min[0]}, {ith_box.x_min[1]}, {ith_box.x_min[2]}}}"\n' \
                       f'\t\t\txMax="{{{ith_box.x_max[0]}, {ith_box.x_max[1]}, {ith_box.x_max[2]}}}"/>\n'

        self.full_model += f'\t<Geometry>\n{box_all}\t</Geometry>\n\n'

    def add_events(self):
        self.full_model += f'\t<Events\n' \
                           f'\t\tmaxTime="{self.events_data.max_time}">\n' \
                           f'\t\t<PeriodicEvent\n' \
                           f'\t\t\tname="{self.events_data.out_name}"\n' \
                           f'\t\t\ttimeFrequency="{self.events_data.out_freq}"\n' \
                           f'\t\t\ttarget="{self.events_data.out_tar_dir}"/>\n' \
                           f'\t\t<PeriodicEvent\n' \
                           f'\t\t\tname="{self.events_data.solver_names[0]}"\n' \
                           f'\t\t\tforceDt="{self.events_data.solver_dts[0]}"\n' \
                           f'\t\t\tendTime="{self.events_data.solver_end_time[0]}"\n' \
                           f'\t\t\ttarget="{self.events_data.solver_target[0]}"/>\n' \
                           f'\t\t<PeriodicEvent\n' \
                           f'\t\t\tname="{self.events_data.solver_names[1]}"\n' \
                           f'\t\t\tforceDt="{self.events_data.solver_dts[1]}"\n' \
                           f'\t\t\tbeginTime="{self.events_data.solver_begin_time[1]}"\n' \
                           f'\t\t\ttarget="{self.events_data.solver_target[1]}"/>\n' \
                           f'\t</Events>\n\n'

    def add_numerical_methods(self):
        self.full_model += f'\t<NumericalMethods>\n' \
                           f'\t\t<FiniteVolume>\n' \
                           f'\t\t\t<TwoPointFluxApproximation name="{self.flow_discretization}"/>\n' \
                           f'\t\t</FiniteVolume>\n' \
                           f'\t</NumericalMethods>\n\n'

    def add_elements_regions(self):
        cell_element_regions_all = f''
        for ith_elem_region in self.element_regions_data:
            cell_element_regions_all += f'\t\t<CellElementRegion\n' \
                                        f'\t\t\tname="{ith_elem_region.name}"\n' \
                                        f'\t\t\tcellBlocks="{{{", ".join([str(ii) for ii in ith_elem_region.region_name])}}}"\n' \
                                        f'\t\t\tmaterialList="{{rock}}"/>\n'

        self.full_model += f'\t<ElementRegions>\n{cell_element_regions_all}\t</ElementRegions>\n\n'

    def add_constitutive(self):
        self.full_model += f'\t<Constitutive>\n' \
                           f'\t\t<CompressibleSolidConstantPermeability\n' \
                           f'\t\t\tname="rock"\n' \
                           f'\t\t\tsolidModelName="nullSolid"\n' \
                           f'\t\t\tporosityModelName="rockPorosity"\n' \
                           f'\t\t\tpermeabilityModelName="rockPerm"/>\n' \
                           f'\t\t<NullModel\n' \
                           f'\t\t\tname="nullSolid"/>\n' \
                           f'\t\t<PressurePorosity\n' \
                           f'\t\t\tname="rockPorosity"\n' \
                           f'\t\t\tdefaultReferencePorosity="{self.constitutive_data.ref_poro}"\n' \
                           f'\t\t\treferencePressure="{self.constitutive_data.ref_pres}"\n' \
                           f'\t\t\tcompressibility="{self.constitutive_data.compr}"/>\n ' \
                           f'\t\t<ConstantPermeability\n' \
                           f'\t\t\tname="rockPerm"\n' \
                           f'\t\t\tpermeabilityComponents="{{{self.constitutive_data.perm[0]}, {self.constitutive_data.perm[1]}, {self.constitutive_data.perm[2]}}}"/>\n' \
                           f'\t</Constitutive>\n\n'

    def add_field_specifications(self):
        base_field_specifications = lambda rock_params, init_cond, bound_cond, perm_poro, source_data_all: \
            f'\t<FieldSpecifications>\n{rock_params}{init_cond}{bound_cond}{perm_poro}{source_data_all}\t</FieldSpecifications>\n\n'

        rock_parameters_all = f''
        for rock_params in self.rock_parameter_data:
            static_template = f'\t\t\tsetNames="{{ all }}"\n' \
                              f'\t\t\tobjectPath="ElementRegions/{rock_params.region_name}"\n'

            for i in range(rock_params.num_rock_params):
                rock_parameters_all += f'\t\t<FieldSpecification\n' \
                                       f'\t\t\tname="{rock_params.region_name}{rock_params.rock_params_name[i]}"\n' \
                                       f'\t\t\tinitialCondition="1"\n' \
                                       f'{static_template}' \
                                       f'\t\t\tfieldName="{rock_params.rock_params_fieldname[i]}"\n' \
                                       f'\t\t\tscale="{rock_params.rock_params_value[i]}"/>\n'
            rock_parameters_all += f'\n'

        inititial_cond_all = f''
        for init_cond in self.initial_condition_data:
            static_template = f'\t\t\tsetNames="{{ all }}"\n' \
                              f'\t\t\tobjectPath="ElementRegions/{init_cond.region_name}"\n'

            inititial_cond_all += f'\t\t<FieldSpecification\n' \
                                  f'\t\t\tname="{init_cond.region_name}InitialPressure"\n' \
                                  f'\t\t\tinitialCondition="1"\n' \
                                  f'{static_template}' \
                                  f'\t\t\tfieldName="pressure"\n' \
                                  f'\t\t\tscale="{init_cond.pres_val}"/>\n' \
                                  f'\t\t<FieldSpecification\n' \
                                  f'\t\t\tname="{init_cond.region_name}InitialTemp"\n' \
                                  f'\t\t\tinitialCondition="1"\n' \
                                  f'{static_template}' \
                                  f'\t\t\tfieldName="temperature"\n' \
                                  f'\t\t\tscale="{init_cond.temp_val}"/>\n'
            for i in range(self.solvers_data.num_comp):
                inititial_cond_all += f'\t\t<FieldSpecification\n' \
                                      f'\t\t\tname="{init_cond.region_name}Init{init_cond.comp_name[i]}"\n' \
                                      f'\t\t\tinitialCondition="1"\n' \
                                      f'{static_template}' \
                                      f'\t\t\tfieldName="globalCompFraction"\n' \
                                      f'\t\t\tcomponent="{i}"\n' \
                                      f'\t\t\tscale="{init_cond.comp_val[i]}"/>\n'
            inititial_cond_all += f'\n'

        bound_cond_all = f''
        for bound_cond in self.boundary_condition_data:
            bound_cond_all += f'\t\t<FieldSpecification\n' \
                              f'\t\t\tname="{bound_cond.source_name}Pressure"\n' \
                              f'\t\t\tobjectPath="ElementRegions/{bound_cond.region_name}"\n' \
                              f'\t\t\tfieldName="pressure"\n' \
                              f'\t\t\tscale="{bound_cond.pres_val}"\n' \
                              f'\t\t\tsetNames="{{ {bound_cond.source_name} }}"/>\n' \
                              f'\t\t<FieldSpecification\n' \
                              f'\t\t\tname="{bound_cond.source_name}TermTemp"\n' \
                              f'\t\t\tsetNames="{{ {bound_cond.source_name} }}"\n' \
                              f'\t\t\tobjectPath="ElementRegions/{bound_cond.region_name}"\n' \
                              f'\t\t\tfieldName="temperature"\n' \
                              f'\t\t\tscale="{bound_cond.temp_val}"/>\n'

            for i in range(self.solvers_data.num_comp):
                bound_cond_all += f'\t\t<FieldSpecification\n' \
                                  f'\t\t\tname="{bound_cond.source_name}{bound_cond.comp_name[i]}"\n' \
                                  f'\t\t\tsetNames="{{ {bound_cond.source_name} }}"\n' \
                                  f'\t\t\tobjectPath="ElementRegions/{bound_cond.region_name}"\n' \
                                  f'\t\t\tfieldName="globalCompFraction"\n' \
                                  f'\t\t\tcomponent="{i}"\n' \
                                  f'\t\t\tscale="{bound_cond.comp_val[i]}"/>\n'
            bound_cond_all += f'\n'

        perm_poro_functions_all = f''
        for perm_poro_func in self.perm_poro_functions:
            perm_poro_functions_all += f'\t\t<FieldSpecification\n' \
                                       f'\t\t\tname="{perm_poro_func.name}"\n' \
                                       f'\t\t\tcomponent="{perm_poro_func.component}"\n' \
                                       f'\t\t\tinitialCondition="1"\n' \
                                       f'\t\t\tsetNames="{{all}}"\n' \
                                       f'\t\t\tobjectPath="ElementRegions/{perm_poro_func.region_name}"\n' \
                                       f'\t\t\tfieldName="rockPerm_permeability"\n' \
                                       f'\t\t\tfunctionName="{perm_poro_func.fun_name}"\n' \
                                       f'\t\t\tscale="{perm_poro_func.scale}"/>\n'
        perm_poro_functions_all += f'\n'

        source_data_all = f''
        for ith_source in self.source_term_data:
            source_data_all += f'\t\t<SourceFlux\n' \
                               f'\t\t\tname="{ith_source.name}"\n' \
                               f'\t\t\tobjectPath="ElementRegions/{ith_source.region_name}"\n' \
                               f'\t\t\tcomponent="{ith_source.component}"\n' \
                               f'\t\t\tscale="{ith_source.scale}"\n' \
                               f'\t\t\tsetNames="{{{ith_source.source_name}}}"/>\n'

        self.full_model += base_field_specifications(rock_parameters_all, inititial_cond_all,
                                                     bound_cond_all, perm_poro_functions_all, source_data_all)

    def add_table_functions(self):
        table_function_all = f''
        for table_i in self.table_function_data:
            table_function_all += f'\t\t<TableFunction\n' \
                                  f'\t\t\tname="{table_i.name}"\n' \
                                  f'\t\t\tinputVarNames="{{elementCenter}}"\n' \
                                  f'\t\t\tcoordinateFiles="{{xlin.geos,ylin.geos,zlin.geos}}"\n' \
                                  f'\t\t\tvoxelFile="{table_i.filename}"\n' \
                                  f'\t\t\tinterpolation="{table_i.interpolation}"/>\n'

        self.full_model += f'\t<Functions>\n{table_function_all}\t</Functions>\n\n'

    def add_outputs(self):
        self.full_model += f'\t<Outputs>\n' \
                           f'\t\t<VTK\n' \
                           f'\t\t\tname="vtkOutput"/>\n' \
                           f'\t</Outputs>\n'

    def write_to_file(self):
        with open(self.file_name, 'w', encoding='utf-8') as f:
            f.write(self.full_xml_file)


class SolversData:
    def __init__(self, num_comp, num_phases, log_level=1, energy_balance=0, max_comp_frac_change=1,
                 trans_mult_exp=3, time_step_cut_fac=0.5, newton_tolerance=1e-4, newton_max_iters=25,
                 direct_parallel=0, use_darts_norm=0):
        self.num_comp = num_comp
        self.num_phases = num_phases
        self.log_level = log_level
        self.energy_balance = energy_balance
        self.max_comp_frac_change = max_comp_frac_change
        self.trans_mult_exp = trans_mult_exp
        self.time_step_cut_fac = time_step_cut_fac
        self.newton_tolerance = newton_tolerance
        self.newton_max_iters = newton_max_iters
        self.direct_parallel = direct_parallel
        self.use_darts_norm = use_darts_norm


class MeshData:
    def __init__(self, name_mesh='mesh1', element_types='C3D8', x_coords=(0, 1000), y_coords=(0, 10), z_coords=(0, 10),
                 nx=(1000), ny=(1), nz=(1), cell_block_name='block1'):
        self.name_mesh = name_mesh
        self.element_types = element_types
        self.x_coords = x_coords
        self.y_coords = y_coords
        self.z_coords = z_coords
        if isinstance(nx, (list, tuple, np.ndarray)):
            self.nx = nx
        else:
            self.nx = [nx]
        if isinstance(ny, (list, tuple, np.ndarray)):
            self.ny = ny
        else:
            self.ny = [ny]
        if isinstance(nz, (list, tuple, np.ndarray)):
            self.nz = nz
        else:
            self.nz = [nz]
        self.cell_block_name = cell_block_name


class GeometryData:
    def __init__(self, x_min=(-0.01, -0.01, -0.01), x_max=(1.01, 10.01, 10.01), name='source'):
        self.x_min = x_min
        self.x_max = x_max
        self.name = name


class EventsData:
    def __init__(self, max_time=8.64e8, out_name='outputs', out_freq=1e7, out_tar_dir='/Outputs/vtkOutput',
                 solver_names=('solver_1', 'solver_2'), solver_dts=(1e2, 5e6), solver_end_time=(1e3, 8.64e8),
                 solver_begin_time=(0, 1e3), solver_target=('/Solvers/compflow', '/Solvers/compflow')):
        self.max_time = max_time
        self.out_name = out_name
        self.out_freq = out_freq
        self.out_tar_dir = out_tar_dir
        self.solver_names = solver_names
        self.solver_dts = solver_dts
        self.solver_end_time = solver_end_time
        self.solver_begin_time = solver_begin_time
        self.solver_target = solver_target


class ConstitutiveData:
    def __init__(self, ref_poro=1.0, ref_pres=0.0, compr=1.0e-7, perm=(3.7e-12, 3.7e-12, 3.7e-13)):
        self.ref_poro = ref_poro
        self.ref_pres = ref_pres
        self.compr = compr
        self.perm = perm


class InitialConditionData:
    def __init__(self, pres_val, temp_val, comp_val, comp_name, region_name):
        self.region_name = region_name
        self.comp_name = comp_name
        self.pres_val = pres_val
        self.temp_val = temp_val
        self.comp_val = comp_val


class BoundaryConditionData(InitialConditionData):
    def __init__(self, pres_val, temp_val, comp_val, comp_name, region_name, source_name):
        super().__init__(pres_val, temp_val, comp_val, comp_name, region_name)
        self.region_name = region_name
        self.source_name = source_name
        self.comp_name = comp_name
        self.pres_val = pres_val
        self.temp_val = temp_val
        self.comp_val = comp_val


class RockParamData:
    def __init__(self, rock_params_name, rock_params_fieldname, rock_params_value, region_name):
        self.region_name = region_name
        self.rock_params_name = rock_params_name
        self.num_rock_params = len(rock_params_name)
        self.rock_params_fieldname = rock_params_fieldname
        self.rock_params_value = rock_params_value


class FunctionData:
    def __init__(self, name='permyFunc', filename='permy.geos', interpolation='nearest'):
        self.name = name
        self.filename = filename
        self.interpolation = interpolation


class PermPoroFunctionData:
    def __init__(self, name='permx', component=0, fun_name='permxFunc', scale=9.869233e-16, region_name='matrix'):
        self.name = name
        self.component = component
        self.fun_name = fun_name
        self.scale = scale
        self.region_name = region_name


class ElementRegionData:
    def __init__(self, name, region_name):
        self.name = name
        self.region_name = region_name


class SourceTermData:
    def __init__(self, name, region_name, component, scale, source_name):
        self.name = name
        self.region_name = region_name
        self.component = component
        self.scale = scale
        self.source_name = source_name
