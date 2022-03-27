# AMINO - Automatic Mutual Information Noise Omission
An algorithm that reduces a large dictionary of order parameters to a smaller set of order parameters by screening for redundancy using a mutual information based distance metric. Please read and cite this manuscript when using AMINO:
https://pubs.rsc.org/--/content/articlehtml/2020/me/c9me00115h

Overviews of each of the subdirectories in this repository are provided below. For detailed descriptions of the files within the subdirectories, refer to the README's within each subdirectory.

## ModelSystems.ipynb

This file contains two analytical model systems that were used for initial testing of AMINO. The input order parameters used are generated internally, rather than from a COLVAR as it would in a real system. It is provided to help users gain an intuition for how the algorithm works on a simple synthetic system.

## BUT.ipynb

Of the provided applications, this is the most similar to a practical use of AMINO. Input data is from a short unbiased MD simulation of the FKBP/BUT protein-ligand system that can be found in the `data` directory. In practice, applications of AMINO should resemble the code provided in this directory.

## amino.py

This is the serial, kernel density estimation Python 3 implementation of AMINO. You should import this into your own projects when you want to use it on your own system.

## amino_fast.py

**This is the version of AMINO that we currently recommend for users.** This is the parallel, kernel density estimation Python 3 implementation of AMINO. You should import this into your own projects when you want to use it on your own system.

## reproducibility/

This directory contains the histogram-based Python 3 implementation of AMINO which is used in the original paper. This version is no longer recommended but remains available for reproducibility.

## amino_output.py

Usage:

```text
python3 amino_output.py <COLVAR> -n <num> -b <bins>
```

where <COLVAR> is the name of the COLVAR file that you want to reduce. The -n flag can be used to specify a maximum number of order parameters in the reduced COLVAR. If no number is provided as input, the default value is 20 or the total number of order parameters in the provided <COLVAR> (whichever is smaller) if the `-n` flag is omitted.

The default value for bins if the `-b` flag is omitted is 50.

The output contains the names of the order parameters (as they were named in the COLVAR) that AMINO has selected. The output is printed to the command line.

AMINO will use a maximum number of output order parameters of 20, but if you would like to override this requirement, you can use '--override' to use the number of order parameters specified by the -n flag.
