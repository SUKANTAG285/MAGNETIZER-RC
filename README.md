# MAGNETIZER-RC: Total Radio Continuum Luminosity

## Overview

`processed_output.py` processes outputs from the
[MAGNETIZER](https://github.com/luizfelippesr/magnetizer) galaxy magnetic field model and computes the radio continuum observables of simulated galaxies, specifically their total synchrotron luminosity.

The script also incorporates outputs from the
[GALFORM](https://ui.adsabs.harvard.edu/abs/2000MNRAS.319..168C/abstract) semi-analytic model (SAM) of galaxy formation. By combining the information from both models, it generates catalogues of galaxy properties together with their radio synchrotron luminosities at a chosen redshift.

## Inputs

- MAGNETIZER output files containing the magnetic field and galaxy evolution information.
- GALFORM output files containing the corresponding galaxy properties.

## Outputs

The script produces processed galaxy catalogues that include:

- Galaxy physical properties from GALFORM.
- Radio continuum synchrotron luminosities computed from MAGNETIZER.
- Additional quantities required for statistical analyses.

## Applications

The processed catalogues can be used for:

- Studying correlations between galaxy properties and radio luminosity.
- Constructing radio luminosity functions (RLFs) at different redshifts.
- Comparing model predictions with observational data.
- Generating figures and analyses using the scripts provided in the `plot_scripts/` directory.

## Workflow

```text
GALFORM Outputs
       │
       ▼
MAGNETIZER Outputs
       │
       ▼
 processed_output.py
       │
       ▼
 Processed Galaxy Catalogues
       │
       ├── Galaxy Property–Radio Luminosity Correlations
       └── Radio Luminosity Functions (RLFs)
