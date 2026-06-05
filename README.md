# MAGNETIZER-RC: Total Synchrotron Luminosity

## Overview

`processed_output.py` processes outputs from the
[MAGNETIZER](https://github.com/luizfelippesr/magnetizer) and computes the radio continuum observable (e.g. total synchrotron luminosity) for simulated galaxies.

The script also incorporates outputs from the
[GALFORM](https://ui.adsabs.harvard.edu/abs/2000MNRAS.319..168C/abstract) semi-analytic model (SAM) of galaxy formation, which provides the galaxy catalogues and associated physical properties used as inputs to MAGNETIZER. By combining the outputs from both models, it generates catalogues of galaxy properties together with their synchrotron luminosities at any chosen redshift up to \(z=3\).

## Inputs

- GALFORM output files containing galaxy catalogues and their physical properties.
- MAGNETIZER output files containing the corresponding magnetic field evolution and related quantities.

## Outputs

The script computes the radio continuum synchrotron luminosity and generates processed galaxy catalogues containing:

- Galaxy physical properties from GALFORM.
- Magnetic-field quantities from MAGNETIZER.
- Computed radio continuum luminosities.

## Applications

The processed catalogues can be used to perform statistical analyses using the scripts provided in the `plot_scripts/` directory, including:

- Studying correlations between galaxy properties and radio luminosity.
- Constructing radio luminosity functions (RLFs) at different redshifts.
- Comparing model predictions with observational data.
  
## Workflow

```text
GALFORM Outputs -----------\
                            \
                             +--> processed_output.py
                            /        |
MAGNETIZER Outputs --------/         |
                                     |
                                     |
                                     v
                        Processed Galaxy Catalogues
                                |
                                +--> Galaxy Property–Radio Luminosity Correlations
                                |
                                +--> Radio Luminosity Functions (RLFs)
```


