# OBL-Table-and-XML-Generator-for-GEOSX
Python scripts for generating OBL tables and XML-files used for simulating multiphase reactive flow using GEOSX platform (https://github.com/GEOSX/GEOSX). DARTS is required for generating the OBL tables. 

To install darts, use the following code (NOTE: it might be easiest to start with a clean virtual environment, see https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html for more information):

```pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple dartsim```

Furthermore, several additional packages are required, all can be installed with one simple command:

```pip install -r requirements.txt```

Several examples are given for generating XML-files (e.g., generate_xml_geosx_1D_benchmark.py). Currently there are three models which are added to this repository: binary compositional model, simple kinetics, and multiphase kinetics (i.e., generate_obl_operators_geosx.py these three cases are used). Other models are going to be added in the future. There is a limited to the number of supporting points that can currently be used. A four components system with 32-42-42-42 points has been tested, meaning 2.371 million control volumes in the (p, z1, z2, z3)-parameter space.

Feel free to experiment with the physics currently available, or modify the current files to suit your needs in terms of required physics. benchmark_model/model.py is where most of the parameters used in the model are specified. physics_sup/operator_evaluator_sup is where the properties are computed and assigned to the operators. 
