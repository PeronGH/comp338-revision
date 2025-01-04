# Low-Level Vision

## Image Formation

### Homogeneous Coordinates

- Add a scaling factor to cartesian coordinates
- Cartesian to homogeneous: $(x, y, z) \rArr \begin{bmatrix} x\\y\\z\\1 \end{bmatrix}$
- Homogeneous to cartesian:  $\begin{bmatrix} x\\y\\2 \end{bmatrix} \rArr (\frac x 2, \frac y 2)$

### 3D-2D Projection

$$
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
$$

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