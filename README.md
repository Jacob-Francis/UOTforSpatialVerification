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

The data for the ICP cases is available from the [ICP project page](https://projects.ral.ucar.edu/icp/) under the ICP tab and the "MesciVICT Cases" section.

## To-Do

- Code to generate 10-increment C1 tests will be added here.
- A notebook for running examples for each dataset, along with corresponding diagrams, will also be added here at a later date.
