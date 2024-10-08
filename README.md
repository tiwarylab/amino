# AMINO - Automatic Mutual Information Noise Omission
An algorithm that reduces a large dictionary of order parameters to a smaller set of order parameters by screening for redundancy using a mutual information based distance metric. Please read and cite this manuscript when using AMINO:
https://doi.org/10.1039/C9ME00115H

Overviews of each of the subdirectories in this repository are provided below.

## ModelSystems.ipynb

This file contains two analytical model systems that were used for initial testing of AMINO. The input order parameters used are generated internally, rather than from a COLVAR as it would in a real system. It is provided to help users gain an intuition for how the algorithm works on a simple synthetic system.

## BUT.ipynb

Of the provided applications, this is the most similar to a practical use of AMINO. Input data is from a short unbiased MD simulation of the FKBP/BUT protein-ligand system that can be found in the `data` directory. In practice, applications of AMINO should resemble the code provided in this directory.

## amino.py

This is the kernel density estimation Python 3 implementation of AMINO. You should import this into your own projects when you want to use it on your own system.

## reproducibility/

This directory contains the histogram-based Python 3 implementation of AMINO which is used in the original paper. This version is no longer recommended but remains available for reproducibility.

## amino_cli.py

Usage:

Command line:
```bash
python amino_cli.py <COLVAR> --n <num> --bins <bins>
python amino_cli.py --help
```

Python or notebook:
```python
import amino_cli
amino_cli.main("<COLVAR>", n=20)
```

`<COLVAR>` is a mandatory argument providing the name of the COLVAR file that you want to reduce. 

The `--n` option can be used to specify a maximum number of order parameters in the reduced COLVAR. Default value is 20 or the total number of order parameters in the provided <COLVAR> (whichever is smaller). AMINO will use a maximum number of output order parameters of 20, but you can use `--override` to remove this limitation.

The `--bins` option specifies how many bins are used when calculating the mutual information and entropy. Default value is 50.

The output contains the names of the order parameters (as they were named in the COLVAR) that AMINO has selected. The output is printed to the command line.
