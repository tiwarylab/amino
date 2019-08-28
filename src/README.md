# src

This directory contains the source code for a Python 3 script that runs AMINO. To use the script, run

```text
python3 amino.py <COLVAR> -o -n <num>
```

where <COLVAR> is the name of the COLVAR file that you want to reduce. The -n flag can be used to specify a maximum number of order parameters in the reduced COLVAR. If no number is provided as input, the default value is 30 or the total number of order parameters in the provided <COLVAR> (whichever is smaller).

The output contains the names of the order parameters (as they were named in the COLVAR) that AMINO has selected. The output is printed to the command line.

AMINO will use a maximum number of output order parameters of 30, but if you would like to override this requirement, you can use '--override' to use the number of order parameters specified by the -n flag.
