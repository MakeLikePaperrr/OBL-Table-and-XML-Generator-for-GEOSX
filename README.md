# OBL-Table-and-XML-Generator-for-GEOSX
Python scripts for generating OBL tables and XML-files used for simulating multiphase reactive flow using GEOSX platform. DARTS is required for generating the OBL tables. 

To install darts, use the following code:

pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple dartsim

Several examples are given for generating XML-files (e.g., generate_xml_geosx_1D_benchmark.py). Currently there are three models which are added to this repository: binary compositional model, simple kinetics, and multiphase kinetics (i.e., generate_obl_operators_geosx.py these three cases are used). Other models are going to be added in the future.
