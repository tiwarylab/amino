import typer
from typing import Annotated
import numpy as np
import amino

def main(filename: Annotated[str, typer.Argument(help="Name of the COLVAR file containing the order parameters")], 
         n: Annotated[int, typer.Option(help="Number of order parameters to be calculated")] = None, 
         bins: Annotated[int, typer.Option(help="Number of bins")] = 50, 
         kde_bandwidth: Annotated[float, typer.Option(help="Bandwidth for the KDE")] = 0.02,
         override: Annotated[bool, typer.Option(help="By default the --n parameter is capped at 20. \
                                                Using --override will default --n option to the number of OPs in the input COLVAR file")] = False):

    trajs = {}

    # Get the names of the order parameters from line 1, skipping first three: #! FIELDS time
    with open(filename) as colvar:
        names = colvar.readline().split()[3:]

    # Skip first column for time and read the rest of the file
    time_series = np.loadtxt(filename).T[1:]

    # Sanity check
    assert(len(names) == time_series.shape[0])

    for i, op in enumerate(names):
        trajs[op] = time_series[i]

    # number of order parameters
    if n is None:
        n = len(names)
    if n > 20 and not override:
        n = 20

    # initializing objects and run the code
    ops = [amino.OrderParameter(i, trajs[i]) for i in names]
#    final_ops = amino.find_ops(ops, n, bins)
    final_ops = amino.find_ops(ops, n, bins, bandwidth=kde_bandwidth)

    print(f"\n{len(final_ops)} AMINO Order Parameters:")
    for i in final_ops:
        j, k = i.name.split('_')[1:]
        print(f"distance :{j}@CA :{k}@CA")

if __name__ == "__main__":
    typer.run(main)
