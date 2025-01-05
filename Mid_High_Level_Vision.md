# Mid & High Level Vision

## Histogram Features

- **Computation**: count number of pixels having each intensity.
- **Normalisation**: divide each count by total number of pixels.
- List a table with 3 columns: "Intensity", "Count/Histogram", "Normalised Histogram".

- **Invariant** to rotation, scaling, mirroring, skew

## LBP Descriptors 

- **Computation**: For each pixel whose intensity is $C$:
  - $\begin{bmatrix}b_1&b_2&b_3\\b_8&C&b_4\\b_7&b_6&b_5\end{bmatrix}$
  - `b_i = (int) b_i >= C`
  - Output: `b1 b2 b3 b4 b5 b6 b7 b8` (8-bit number)
    - Can be converted to decimal

- **Invariant** to brightness change.

- May **collide**: different regions may have the same LBP feature.

## kNN (K Nearest Neighbour)

- Supervised

- For a new point:
  1. Find $k$ closest points by L2 distance
  2. Let the closet points vote for the category of the new point.