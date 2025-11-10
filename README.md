# UOTforSpatialVerification

Unbalanced Optimal Transport implementation with total variation or Kullback-Leibler marginal penalization for precipitation forecasts verification.

## Requirementss
- `torch`
- `geomloss`
- `pykeops`
- `python3.13>=>python3.8`

There may be a lot of deprecation warnings with python3.13

Installing `geomloss` will automatically install the necessary dependencies.

## Setting up Venv

Once you have cloned your the git repo, you can create a new python vevn using;
```bash
python3 -m venv uotvenv
```
This will creat a new virtual environment called `uotvenv`. To activate this and use this python environment use

```bash
source uotvenv/bin/activate
```
New to your commandline prompt you shoudl see `(uotvenv)` appear, meaning that this venv is actiavted.
Then comand `which python` will also double check that it points to the collect python inside the uotvenv folder. 

With the venv activated you can start the installation.

## Installation

You may need to update pip and setuptools using;
```bash
pip install --upgrade pip setuptools wheel
```

Then you can install all the classes in editable mode by, moving into the root folder (`cd UOTforSpatialVerification`) and running:
```bash
bash install_all.bash
```
Alternatively manually install each through
```bash
python3 -m pip install -e torch_numpy_process/ 
python3 -m pip install -e flipflop/ 
python3 -m pip install -e tensoring/  
python3 -m pip install -e unbalanced_ot_metric/ 
```

You can test the installation by running:

```bash
bash testing_ball.bash
```
Or similarly line by line
```bash
pytest -x torch_numpy_process/tests/
pytest -x tensoring/tests/
pytest -x unbalanced_ot_metric/tests/
```

## Data

The data for the ICP cases is available from the [ICP project page](https://projects.ral.ucar.edu/icp/) under the ICP tab and the "MesoVICT Cases" section.

- The circular, ellipse and other simple geometric cases can be found under the ICP tab MesoVICT tab, or through https://projects.ral.ucar.edu/icp/Data/NewGeom/test_fields.zip

- The Perturbed (fake00X) and spring 2005 cases as well as the two-step intensity data (which wasn't replciated in our work) is avaliable under the same "ICP tab", and through https://projects.ral.ucar.edu/icp/Data/Cases20081023.tar.gz . Note that the true Lon Lat daat isavaliable in the spring2005 subfile and correspond to all the data s they are on the same mesh.

- The VERA analysis and assocaied core case as avaliable again under the "MesoVICT Cases" tab or through 

-- VERA: https://projects.ral.ucar.edu/icp/Data/VERA/VERA_case1.zip
-- CMH: https://projects.ral.ucar.edu/icp/Data/CMH/CMH_06_case1.tar.gz
-- CO2: https://projects.ral.ucar.edu/icp/Data/COSMO2.00/CO2_00_case1.tar.gz

