# AMINO - Automatic Mutual Information Noise Omission
An algorithm that reduces a large dictionary of order parameters to a smaller set of order parameters by screening for redundancy using a mutual information based distance metric. [Here](https://www.biorxiv.org/content/biorxiv/early/2019/08/24/745968.full.pdf) is a link to the biorxiv paper.

Overviews of each of the subdirectories in this repository are provided below. For detailed descriptions of the files within the subdirectories, refer to the README's within each subdirectory.

## ModelSystems

This directory contains two analytical model systems that were used for initial testing of AMINO. The code for the algorithm is the same as for experimental systems, but the input order parameters used are generated internally, rather than from a COLVAR as it would in a real system. It is provided to help users gain an intuition for how the algorithm works.

## BUT

Of the provided applications, this is the most similar to a practical use of AMINO. Input data is from a short unbiased MD simulation of the FKBP/BUT protein-ligand system. In practice, applications of AMINO should resemble the code provided in this directory. It is important to note that since this is a realistic application, a single run can take around 20-30 minutes, depending on your system.

## src

This is the final, ready-to-use version of AMINO. In this directory, there is a script that takes in a COLVAR and outputs the order parameters selected by AMINO, as well as a complete Python 3 implementation of all of the functions used in AMINO.

## amino.py
This is the Python 3 file mentioned above (from the src directory) that contains the function definitions used in AMINO.
