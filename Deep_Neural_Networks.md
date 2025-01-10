# Deep Neural Networks

## CNN Size

- Previous layer: $W_1 \times H_1 \times D_1$
- 4 hyperparameters:
  - Number of filters $K$
  - Filter size $F$
  - The stride $S$
  - The amount of zero padding $P$
- Next layer: $W_2 \times H_2 \times D_2$
  - $W_2 = \lfloor \frac{W_1-F+2P}{S} \rfloor + 1$
  - $H_2 = \lfloor \frac{H_1-F+2P}{S} \rfloor +1$
  - $D_2 = K$

## CNN Layers

- **Convolutional layers** extract features using learnable filters.
- **Activation functions** introduce non-linearity.
- **Pooling layers** reduce dimensionality, which improves computational efficiency and add some translation invariance.
- **Fully connected layers** combine features for classification or regression.
- **Deconvolution layers** upsample feature maps. (often used in image segmentation to restore the spatial dimensions)