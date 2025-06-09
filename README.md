# UOTforSpatialVerification

Unbalanced Optimal Transport implementation with total variation or Kullback-Leibler marginal penalization for precipitation forecasts verification.

## Requirementss
- `torch`
- `geomloss`
- `pykeops`

Installing `geomloss` will automatically install the necessary dependencies.

## Installation

You can install all the classes in editable mode by running:

```bash
bash install_all.bash
```

You can test the installation by running:

```bash
bash testing_ball.bash
```

## Data

The data for the ICP cases is available from the [ICP project page](https://projects.ral.ucar.edu/icp/) under the ICP tab and the "MesoVICT Cases" section.

- The circular, ellipse and other simple geometric cases can be found under the ICP tab MesoVICT tab, or through https://projects.ral.ucar.edu/icp/Data/NewGeom/test_fields.zip

- The Perturbed (fake00X) and spring 2005 cases as well as the two-step intensity data (which wasn't replciated in our work) is avaliable under the same "ICP tab", and through https://projects.ral.ucar.edu/icp/Data/Cases20081023.tar.gz . Note that the true Lon Lat daat isavaliable in the spring2005 subfile and correspond to all the data s they are on the same mesh.

- The VERA analysis and assocaied core case as avaliable again under the "MesoVICT Cases" tab or through 

-- VERA: https://projects.ral.ucar.edu/icp/Data/VERA/VERA_case1.zip
-- CMH: https://projects.ral.ucar.edu/icp/Data/CMH/CMH_06_case1.tar.gz
-- CO2: https://projects.ral.ucar.edu/icp/Data/COSMO2.00/CO2_00_case1.tar.gz

## running Test Cases

## Plotting figures


## To-Do

- Code to generate 10-increment C1 tests will be added here.
- A notebook for running examples for each dataset, along with corresponding diagrams, will also be added here at a later date.
