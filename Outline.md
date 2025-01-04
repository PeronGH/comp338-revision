# Outline

## I. Introduction to Computer Vision

*   **Module Administration:**
    *   Module Staff and Contact Information
    *   Module Delivery (Lectures, Labs, Office Hours)
    *   Assessment (Exams, Projects)
    *   Syllabus Overview (Low-level, Mid-level, High-level Vision, Deep Learning)
    *   Additional Resources (Textbooks, Online Courses)
*   **What is Computer Vision?**
    *   Definitions of Computer Vision (Marr, Trucco & Verri, Stockman & Shapiro, Forsyth & Ponce)
    *   Challenges in Computer Vision (Viewpoint, Illumination, Noise, Occlusion, etc.)
    *   Equivalent Names (Machine Vision, Image Analysis, etc.)
    *   Relationship with other fields (AI, Robotics, NLP, etc.)
*   **Applications of Computer Vision**
    *   Sensors (Camera, Lidar, Radar, CT)
    *   Character Recognition (OCR, ANR)
    *   Face Recognition
    *   Image Retrieval
    *   3D Models from Images
    *   Medical Imaging
    *   Autonomous Vehicles
    *   Industrial Inspection, Robot Navigation, Surveillance, etc.
    *   Current Trends and Job Opportunities

## II. Low-Level Vision

*   **Image Formation**
    *   Light Source, Surface Properties, and Sensors
    *   Pinhole Camera Model
    *   World, Camera, Image, and Pixel Coordinates
    *   Homogeneous vs. Cartesian Coordinates
    *   Rotation and Translation Matrices
    *   Perspective Camera Model
    *   Intrinsic and Extrinsic Camera Parameters
*   **Digitalization**
    *   Pixels and Digital Images (2D Array Representation)
    *   Image Origin and Coordinate System
    *   Pixel Values (Grayscale, Color)
*   **Image Representation**
    *   Binary Images
    *   Grayscale Images
    *   Color Images (RGB, HSV, CMYK)
    *   Switching Between Representations (RGB to Grayscale, Grayscale to Binary)
*   **Image Manipulation**
    *   Shrinking/Sub-sampling (Resolution, Under-sampling)
    *   Zooming/Up-sampling (Resolution, Interpolation)
    *   Geometric Transformations (Translation, Scaling, Rotation)
*   **Image Filtering and Convolution**
    *   Cross-Correlation vs. Convolution
    *   2D Image Convolution
    *   Padding and Stride
    *   Example Filters/Masks (Identity, Shift, Box Blur)
*   **Edge Detection**
    *   Origins of Edges (Discontinuities)
    *   Derivatives in 1D and 2D
    *   Discrete Derivative Filters (Backward, Forward, Central)
    *   Gradient Magnitude and Direction
    *   Edge Operators (Roberts, Sobel, Laplacian)
    *   Canny Edge Detector (5 Steps)

## III. Mid-Level Vision

*   **Feature Extraction**
    *   Why Features are Needed (Invariance, Correspondence Problem)
    *   Histogram Features
        *   Definition and Calculation
        *   Normalization
        *   Applications and Properties
    *   LBP (Local Binary Pattern) Descriptors
        *   Calculation and Properties
        *   Pros and Cons
*   **Image Segmentation**
    *   Supervised, Un-supervised, Semi-supervised Segmentation
    *   Thresholding
        *   Regions based on Brightness Differences
        *   Choosing Thresholds
        *   Limitations
    *   Region-Based Methods
        *   Region Growing
        *   K-means Clustering
    *   Edge-Based Methods

## IV. High-Level Vision

*   **Image Classification**
    *   Assigning Labels to Images
    *   Challenges (Viewpoint, Illumination, Scale, Deformation, Occlusion, etc.)
    *   Traditional vs. Machine Learning Approaches
    *   Datasets (MNIST, CIFAR, ImageNet)
*   **Machine Learning for Image Classification**
    *   Data Collection and Labeling
    *   Training a Classifier
    *   Evaluation on New Images
    *   k-Nearest Neighbors (k-NN)
    *   Distance Metrics (L1, L2)
*   **Parametric Approach**
    *   Linear Classifier
    *   Loss Function (e.g., SVM Loss, Cross-Entropy Loss)
    *   Optimization (Gradient Descent)
*   **Object Detection**
    *   Problem Definition (Classification + Localization)
    *   Sliding Window Approach
    *   Region Proposal Approach (R-CNN, Fast R-CNN, Faster R-CNN)
    *   YOLO (You Only Look Once)
    *   Intersection over Union (IoU)
    *   Non-Maximum Suppression (NMS)

## V. Basics of Deep Learning

*   **Artificial Neural Networks**
    *   Neurons and Connections
    *   Activation Functions (Sigmoid, ReLU, etc.)
    *   Linear Separability
    *   Multi-layered Networks
    *   Feed-Forward Neural Networks
*   **Backpropagation**
    *   Gradient Descent
    *   Computing Gradients (Chain Rule)
    *   Updating Weights
*   **Training Neural Networks**
    *   Training Pipeline
    *   Loss Functions
    *   Optimizers
    *   Learning Rate
    *   Batch Size, Epochs
    *   Overfitting and Underfitting
*   **Regularization**
    *   Training, Validation, and Test Sets
    *   L1 and L2 Regularization
    *   Dropout
    *   Early Stopping
    *   Data Augmentation

## VI. Deep Learning for Computer Vision

*   **Convolutional Neural Networks (CNNs)**
    *   Convolutional Layers
    *   Pooling Layers
    *   Fully Connected Layers
    *   Activation Functions (ReLU, etc.)
    *   Batch Normalization
    *   Dropout
*   **Typical Deep Models**
    *   LeNet-5
    *   AlexNet
    *   VGG-16
    *   GoogLeNet (Inception)
    *   ResNet (Residual Networks)
    *   DenseNet
    *   SENet (Squeeze-and-Excitation Networks)
    *   MobileNet
*   **Transfer Learning**
    *   Definition and Benefits
    *   Fine-tuning Pre-trained Models
    *   Applications
*   **Metric Learning**
    *   Learning Distance Functions
    *   Contrastive Learning (Loss)
    *   Applications (Face Verification, etc.)
*   **Object Segmentation with Deep Learning**
    *   Semantic Segmentation
    *   Fully Convolutional Networks (FCN)
    *   UNet
    *   SegNet
    *   PSPNet
    *   Instance Segmentation
    *   Mask R-CNN
*   **Generative Models**
    *   Generative Adversarial Networks (GANs)
    *   Variational Autoencoders (VAEs)
    *   Diffusion Models
    *   Applications (Image Generation, Text-to-Image, etc.)

## VII. Deep Learning Frameworks

*   **Overview of Frameworks**
    *   TensorFlow
    *   PyTorch
    *   Keras
    *   Caffe
    *   Theano
*   **PyTorch Basics**
    *   Tensors vs. NumPy Arrays
    *   Basic Operations (Addition, Multiplication, etc.)
    *   Building Simple Networks
    *   Training and Evaluation
*   **Open Source Resources**
    *   OpenMMLab (MMDetection, etc.)
    *   Detectron2
