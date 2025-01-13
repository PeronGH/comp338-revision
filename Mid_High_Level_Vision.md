# Mid & High Level Vision

## Histogram Features

- **Computation**: count number of pixels having each intensity.
- **Normalisation**: divide each count by total number of pixels.
- List a table with 3 columns: "Intensity", "Count/Histogram", "Normalised Histogram".

- **Invariant** to rotation, scaling, mirroring, skew

## LBP Descriptors 

- **Computation**: For each pixel whose intensity is $C$:
  - $\begin{bmatrix}b_1&b_2&b_3\\b_8&C&b_4\\b_7&b_6&b_5\end{bmatrix}$
  - `b_i = (int) (b_i >= C)`
  - Output: `b1 b2 b3 b4 b5 b6 b7 b8` (8-bit number)
    - Can be converted to decimal

- **Invariant** to brightness change.

- May **collide**: different regions may have the same LBP feature.

## kNN (K Nearest Neighbour)

- Supervised

- For a new point:
  1. Find $k$ closest points by **L2** distance
  2. Let the closet points vote for the category of the new point.

## K-means Clustering

- Unsupervised
- Steps:
  1. Pick $k$ random points as cluter centre
  2. Allocate each point to closet centre (by L2 distance)
  3. Set new centre points to the mean position in that cluster
  4. Go to step 1 unless the centre points are not changed
- Limitations:
  - Need to pick $k$ random points
  - Local optimum

## SIFT

- Steps:
  1. Determine approximate location and scale of keypoints
     - By looking at intensity changes of difference between Gaussians at 2 nearby scales
     - Contribute to scale invariance
  2. Refine keypoint
     - By removing noisy, low-contrast or edge keypoints
  3. Assign orientations to each keypoint
     - Contribute to rotation invariance
  4. Determine descriptors for each keypoint
     - 128-dimensional feature vector

- Pros:
  - Very robust, invariant to scaling and rotation
  - Can handle changes in viewpoint
  - Can handle significant changes in illumination
  - Fast and efficient

- Cons:
  - Not fully invariant
  - Still more complex than simper ones

## SVM Loss

- For $i_{\text{th}}$ example and $j_{\text{th}}$ class:

  - $L_i = \sum_{j \neq y_i} \max(0, s_j - s_{y_i} + \Delta)$

  - Where $y_i$ is the index of the correct class, and $\Delta$ is usually set to 1.

- The total loss is the average loss of all examples.

## Forward Propagation

- Similar to Perceptron

## Back Propagation

- **Gradient Descent**: $w_\text{new} = w_\text{old} - \alpha  \frac{\partial L}{\partial w}$
  - By default, learning rate $\alpha = 1$
  - For $y = wx+b$:
    - By **chain rule**, $\frac{\partial L}{\partial w} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial w}$
    - $\frac{\partial L}{\partial y} = 1$ or the value of error (based on loss function)
    - $\frac{\partial y}{\partial w} = x$

