# src

This directory contains the source code for a Python 3 script that runs OPAF. To use the script, run

```text
python3 opaf.py <COLVAR> -o <NEW_FILE> -n <num>
```

where <COLVAR> is the name of the COLVAR file that you want to reduce and <NEW_FILE> is the name of the file where you want the "reduced" COLVAR to be output. The -n flag can be used to specify a maximum number of order parameters in the reduced COLVAR. If no number is provided as input, the default value is the total number of order parameters in the provided <COLVAR>.
