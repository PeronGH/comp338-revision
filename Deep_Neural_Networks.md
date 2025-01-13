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
- **Activation functions** improve the non-linearity representation ability of the network.
- **Pooling layers** reduce the feature size and keep the important information.
- **Fully connected layers** mix and enhance features.
- **Batch normalisation layers** stabilise the learning process and reduce number of epochs required for training.
- **Dropout layers** improve the generalisation ability of the model and overcome overfitting problem.
- **Deconvolution (transposed convolution) layers** upsample feature maps. (often used in image segmentation to restore the spatial dimensions)

## Training-Validation-Test Set

- **Training set**: Used to train the model.
- **Validation set**: Used to tune hyperparameters and monitor for overfitting during training.
- **Test set**: Used to evaluate the final performance of the trained model on unseen data.

- **Typical split:**
  - **Training set vs validation set:** 80% vs 20% or 90% vs 10%
  - **Training vs validation vs testing set:** 60%-20%-20%

## Handling Overfitting and Underfitting

- **Underfitting:**

  - Use larger network, e.g., more layers, more hidden units

  - Train the network with more epochs or iterations

  - Design a new network architecture

- **Overfitting:**

  - Use more training data

  - Add regularization

  - Design a new network architecture

## Regularisation

- Techniques used to prevent overfitting, where the model performs well on training data but poorly on unseen data.
- Common regularization techniques include:
  - **L1/L2 Regularization**: Adds a penalty term to the loss function based on the magnitude of the weights.
  - **Dropout**: Randomly deactivates neurons during training, forcing the network to learn more robust features.
  - **Early Stopping**: Monitors performance on a validation set and stops training when the performance starts to degrade.
  - **Data Augmentation**: Increases the size of the training dataset by applying transformations to existing data (e.g., rotation, scaling, flipping).
  - **Batch Normalization**: Normalizes the activations within each layer, stabilizing training.

## Network Training Pipeline

1. **Sample** a batch of data.
   - **Data processing:** normalisation, augmentation
   - Split training set into **batches**.
2. **Forward Propagation**.
   - Initialise weights
   - Set up network structure
   - Set up loss function 
3. **Backward Propagation**.
4. **Optimisation** (update parameters).

## Loss Function

### Cross-Entropy Loss

- For classification
- In PyTorch, implictly includes softmax

$$
\text{CrossEntropyLoss} = - \frac{1}{n} \sum_{i=1}^n y_i \log(\hat{y}_i)
$$

Where:

- $n$: number of classes
- $y_i$: ground truth for class $i$ (1 for true, 0 for false)
- $\hat{y}_i$: predicted probability for class $i$

### Mean Squared Error

- For regression

$$
\text{MSE} = \frac{1}{m} \sum_{i=1}^m (y_i - \hat{y}_i)^2
$$

Where:

- $y_i$: ground truth for the $i_\text{th}$ example
- $\hat{y}_i$: predicted probability for the $i_\text{th}$ example

## Optimiser

- Common optimisers: Gradient descent, Stochastic gradient descent (SGD), Mini-batch gradient descent

- **Learning rate:**

  - With a small learning rate, updates to the weights are small, which will guide the optimizer gradually towards the minima. However, the optimizer **may take too long to converge or get stuck in an undesirable local minima**;

  - With a large learning rate, the algorithm **learns fast**, but it **may also cause the algorithm to oscillate around or even jump over the minima**. Even worse, a high learning rate equals large weight updates, which might **cause the weights to overflow, not converge**;

  - A good learning rate is a tradeoff between the coverage rate and overshooting. It’s not too small so that our algorithm can converge swiftly, and it’s not too large so that our algorithm **won’t jump back and forth without reaching an undesirable local minima**.

## Activation Function

### ReLU

* **Formula:**  $f(x) = max(0, x)$  
* **Graph:** A straight line at zero for all negative inputs, and a straight line with slope 1 for all positive inputs. It essentially "rectifies" negative values to zero.
* **Explanation:** ReLU replaces all negative input values with zero.  Positive values remain unchanged.

### Sigmoid

* **Formula:**  $f(x) = \frac{1}{1 + e^{-x}}$ also known as the logistic function.
* **Graph:** An 'S'-shaped curve that smoothly transitions from 0 to 1.  Outputs are bound between 0 and 1.
* **Explanation:** Sigmoid squashes any input value (from negative infinity to positive infinity) into the range between 0 and 1.   This is often interpreted as a probability.
* **Derivative:** $\sigma{\prime}(x) = \sigma(x) \cdot (1 - \sigma(x))$

### Softmax

- Converts logits to probabilities for classification

$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}
$$

Where:

- $z_i$: the score for the $i_\text{th}$ example
- $n$: number of examples

## Deep Models

- **Count layers:** Ignore layers without parameters (e.g. pooling, activation)
- **Design preferences:**
  - Small filters, deeper networks
    - Why small filters:
      - Stack of multiple small filters have the same effective receptive field as a larger one
      - Fewer parameters
    - Why deeper:
      - More activation layers, more non-linearity, better model representation
  - Reduce feature map sizes, increase feature map depth
  - Repeat convolution-activation-pooling, and finally fully connected network
  - ResNet: very deep networks using residual connections

- **Object Detection**

  - Sliding window based– one stage
    - YOLO

  - Region proposal based– two stage

    - R-CNN

    - Fast R-CNN

    - Faster R-CNN

- **Object Segmentation**

  - Sliding Window + Classification of each pixel

  - Fully Convolutional Networks (FCN)

    - FCN

    - UNet

    - SegNet

    - PSPNet

  - Instance Segmentation
    - Mask RCNN

## Transfer learning

(Like fine-tuning LLMs)

**Advantages:**

- Provides fast training progress, you don’t have to start from scratch using randomly initialized weights

- You can use small training dataset to achieve decent results

## Metric Learning

(Like embedding models)

- Measures similarity
- **Applications:** Face verification, Fingerprint Recognition, Signature Verification, Text Similarity

##  RNN

- **Input:** sequential data
- **Application**: predictions (weather, stock market)
- **Problem:** cannot handle long-term dependencies
  - **Long Short-Term Memory (LSTM)** can address the vanishing gradient problem in standard RNNs.

## GAN

- **Generator:** try to fool the discriminator by generating real-looking images

- **Discriminator:** try to distinguish between real and fake images

- **Application**: image generation, video generation

- **Training:**

$$
\min_{\theta_g} \max_{\theta_d} \left[ \mathbb{E}_{x \sim p_{\text{data}}} \log D_{\theta_d}(x) + \mathbb{E}_{z \sim p(z)} \log \left(1 - D_{\theta_d}(G_{\theta_g}(z))\right) \right]
$$

Where:

- $D(x)$ is the discriminator's estimate of the probability that real data instance $x$ is real.

- $\mathbb{E}_x$ is the expected value over all real data instances.

- $G(z)$ is the generator's output when given noise $z$.

- $D(G(z))$ is the discriminator's estimate of the probability that a fake instance is real.

- $E_z$ is the expected value over all random inputs to the generator (in effect, the expected value over all generated fake instances $G(z)$).