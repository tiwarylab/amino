# ModelSystems

This directory contains files for AMINO runs on two analytical model systems. All of the order parameters in both of these systems fall in one of the following two categories:

1. A parameter sampled randomly from the range [-1, 1]. The value of the parameter at each point in the time series is independent from the values at other points in the time series. Naming convention: capital letter (single character)
2. A parameter derived from one of the above parameters with noise added. The noise is in the range [-0.05, 0.05]. Naming convention: capital letter of the original order parameter followed by a unique number.

The goal of AMINO is to correctly determine the number of order parameters in the first category and identify which parameters they are.

## model_output.ipynb

This Jupyter notebook file just outputs the names of the order parameters that AMINO selects. The provided code runs the 5 and 10 order parameter model systems. The system with 5 original order parameters begins with a total of 62 order parameters, and the 10 order parameter model system begins with 120 order parameters. Any custom analytical system can be built for testing using the provided helper functions.

## model_detailed.ipynb

This file does the same calculations as model_output.ipynb, but additional output is returned that helps give intuition about the sub-steps within AMINO. The provided output includes a plot showing distortion for different numbers of order parameters, as well as the final grouping of order parameters around the selected order parameters.
