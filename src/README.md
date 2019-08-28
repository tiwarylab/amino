# src

This directory contains the source codes for two Python 3 scripts that run AMINO. These scripts will ignore the first column since this is usually the time column. If you have removed the time column manually (or have modified the COLVAR in some other form), you should modify the script to suit your specific needs.

## amino_output.py

```text
python3 amino_output.py <COLVAR> -n <num>
```

where <COLVAR> is the name of the COLVAR file that you want to reduce. The -n flag can be used to specify a maximum number of order parameters in the reduced COLVAR. If no number is provided as input, the default value is 30 or the total number of order parameters in the provided <COLVAR> (whichever is smaller).

The output contains the names of the order parameters (as they were named in the COLVAR) that AMINO has selected. The output is printed to the command line.

AMINO will use a maximum number of output order parameters of 30, but if you would like to override this requirement, you can use '--override' to use the number of order parameters specified by the -n flag.

## amino.py

This file just contains all of the function definitions used in AMINO (Python 3). You can import this file into whatever Python file you want and use it as you please.
