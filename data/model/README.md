# Figure 5

** NOTE: This is just copied from the BUT file, everything is the same, but with different data (obviously)

## num_op_array.npy
This is just the numbers of order parameters that correspond to the distortions in distortion.npy. The last number is intentionally 2, since the y-axis is the jump in distortion, and distortion is undefined for 0 order parameters.

## distortion.npy
The actual jumps in distortion values for different dimensions. The first element of this array is the jumps in distortion with -0.5 as the exponent, the second is the jumps with -1.0 as the exponent, and so on (exponent increasing by 0.5 for each element).
