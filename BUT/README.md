# BUT

This directory contains files for AMINO runs on an unbiased trajectory of a protein-ligand system. The protein is FKBP, and its ligand is BUT. In the provided COLVAR, the ligand is bound to the protein for the entire trajectory (10 nanoseconds). Here we show that AMINO can be used to bias towards unbinding events using metadynamics. We used these order parameters to generate a reaction coordinate using [SGOOP](https://aip.scitation.org/doi/10.1063/1.5064856). Refer to our [paper](https://www.biorxiv.org/content/biorxiv/early/2019/08/24/745968.full.pdf) for more details.

## BUT-output.ipynb

This Jupyter notebook file just outputs the names of the order parameters that AMINO selected for the protein-ligand system. This is the code that should be used for systems that users are interested in.

## BUT-Debug.ipynb

This file does the same calculations as BUT-output.ipynb, but additional output is returned that is aimed to help users understand the sub-steps within AMINO.
