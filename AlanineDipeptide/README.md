# AlanineDipeptide

This is the directory for OPAF runs on the alanine dipeptide system.

The goal of OPAF is to correctly determine the number of order parameters in the first category and identify which parameters they are.

## model_output.ipynb

This Jupyter notebook file just outputs the names of the order parameters that OPAF selects. The provided code runs the 5 and 10 order parameter model systems. The system with 5 original order parameters begins with a total of 62 order parameters, and the 10 order parameter model system begins with 120 order parameters. Any custom analytical system can be built for testing using the provided helper functions.

## model_detailed.ipynb

This file does the same calculations as model_output.ipynb, but additional output is returned that helps give intuition about the sub-steps within OPAF. The provided output includes a plot showing distortion for different numbers of order parameters, as well as the final grouping of order parameters around the selected order parameters.
