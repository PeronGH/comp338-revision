# Low-Level Vision

## Image Formation

### Homogeneous Coordinates

- Add a scaling factor to cartesian coordinates
- Cartesian to homogeneous: $(x, y, z) \rArr \begin{bmatrix} x\\y\\z\\1 \end{bmatrix}$
- Homogeneous to cartesian:  $\begin{bmatrix} x\\y\\2 \end{bmatrix} \rArr (\frac x 2, \frac y 2)$

### 3D-2D Projection

```math
\begin{bmatrix}
u \\
v \\
1
\end{bmatrix}

=

\frac{1}{Z_c}
\begin{bmatrix}
\frac{1}{dx} & 0 & u_0 \\
0 & \frac{1}{dy} & v_0 \\
0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
f & 0 & 0 & 0 \\
0 & f & 0 & 0 \\
0 & 0 & 1 & 0
\end{bmatrix}
\begin{bmatrix}
R & T \\
\vec{0} & 1
\end{bmatrix}
\begin{bmatrix}
X_w \\
Y_w \\
Z_w \\
1
\end{bmatrix}
```

Usually assume:

- $Z_c = Z_w$
- $\begin{bmatrix} R & T \\ \vec{0} & 1 \end{bmatrix} = \begin{bmatrix}1 &0&0&0\\0&1&0&0\\0&0&1&0\\0&0&0&1\end{bmatrix}$

Where:

*   $\begin{bmatrix} X_w \\ Y_w \\ Z_w \\ 1 \end{bmatrix}$ are world coordinates (homogeneous).

*   $R$ is the rotation matrix.

*   $T$ is the translation vector.
*   $\begin{bmatrix} R & T \\ \vec{0} & 1 \end{bmatrix}$ is **extrinsic camera matrix**.

*   $f$ is the focal length.

*   $u_0$ , $v_0$ are the coordinates of the principal point.

*   $\frac{1}{dx}$ , $\frac{1}{dy}$  are the scaling factors (magnification factors).
*   $(u, v)$ are pixel coordinates.

## Digitalization

- **Origin** $(0,0)$ is the top-left pixel.

## Image Filtering

- Cross-Correlation applies the filter directly.
- Convolution rotates the filter by 180 degrees before applying it.
- **Convolution in neural networks is actually cross-correlation**.

### Different Filters

#### Box Blur

- Example: $\begin{bmatrix} \frac 1 9 & \frac 1 9 & \frac 1 9 \\ \frac 1 9 & \frac 1 9 & \frac 1 9 \\ \frac 1 9 & \frac 1 9 & \frac 1 9 \end{bmatrix}$

#### 1D Discrete Derivate Filters

For the image $\begin{bmatrix}f(x-1)&f(x)&f(x+1)\end{bmatrix}$:

- **Backward filter**:
  - $\begin{bmatrix}-1&1&0\end{bmatrix}$
  - $f(x)-f(x-1)=f'(x)$

- **Forward filter**
  - $\begin{bmatrix}0&1&-1\end{bmatrix}$
  - $f(x)-f(x+1)=f'(x)$

- **Central filter**
  - $\begin{bmatrix}-1&0&1\end{bmatrix}$
  - $f(x+1)-f(x+1)=f'(x)$

#### 2D Discrete Derivate Filters

- Gradient vector: $\nabla f(x,y)=\begin{bmatrix}\frac{\partial f(x, y)}{\partial x}\\\frac{\partial f(x, y)}{\partial y}\end{bmatrix}=\begin{bmatrix}f_x\\f_y\end{bmatrix} $

- Gradient magnitude: $\|\nabla f(x, y)\| = \sqrt{f_x^2 + f_y^2}$

- Gradient direction: $\theta = \tan^{-1}\left(\frac{f_y}{f_x}\right)$

- **Laplacian filter**: $\begin{bmatrix}0&1&0\\1&-4&1\\0&1&0\end{bmatrix}$

  ```math
  \nabla^2f = f_x^2 + f_y^2
  \newline = f(x+1,y)-2f(x,y)+f(x-1,y) + f(x,y+1)-2f(x,y)+f(x,y-1)
  \newline = f(x+1,y)+f(x-1,y)+f(x,y+1)f(x,y-1)-4f(x,y)
  ```

- **Roberts filter**
  - Example: $\begin{bmatrix}0&1\\-1&0\end{bmatrix}$

- **Sobel filter**
  - Example: $\begin{bmatrix}-1&0&1\\2&0&2\\-1&0&1\end{bmatrix}$

