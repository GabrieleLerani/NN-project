## Overview
This directory contains the collection of dataset that can be used for various tasks, including image classification, speech recognition and natural language processing.

- **s_mnist**: A sequential version of the MNIST dataset, designed for tasks involving sequential or time-series processing of images.

- **p_mnist**: The permuted MNIST dataset, a variant of MNIST where the pixels in each image are randomly permuted, making it a challenging benchmark for sequence-based models.

- **cifar10**: The standard CIFAR-10 dataset, containing 60,000 32x32 color images in 10 different classes, with 50,000 training and 10,000 testing images. This dataset is commonly used for image classification tasks.

- **s_cifar10**: A sequential variant of the CIFAR-10 dataset.

- **cifar100**: Similar to CIFAR-10, but with 100 classes grouped into 20 superclasses.

- **stl10**: The STL-10 dataset, which contains 96x96 color images. It is specifically designed for unsupervised learning and has fewer labeled images.

- **speech_mfcc**: This dataset contains Mel Frequency Cepstral Coefficients (MFCCs) extracted from raw speech signals, commonly used for tasks such as speech recognition or speaker identification.

- **speech_raw**: The raw speech waveform dataset, suitable for direct application of 1D CNNs or end-to-end speech recognition models.

- **pathfinder**: The Pathfinder dataset is designed for visual reasoning tasks, particularly involving finding paths between points in noisy visual environments. It is a benchmark for cognitive tasks involving long-range dependencies in images.

- **s_pathfinder**: A sequential version of the Pathfinder dataset, suitable for models that process images as a sequence of steps, typically used in tasks involving attention and sequence processing.

- **listops**: A dataset for natural language processing tasks involving mathematical or logical operations on lists. It is a benchmark for compositional generalization and reasoning tasks.

- **image**: A general-purpose image dataset that can be used for various image classification or segmentation tasks. The exact nature of the images depends on the specific configuration used.

- **s_image**: A sequential version of the general image dataset, allowing images to be processed as a sequence of pixels or patches, suitable for sequence models.

- **text**: A text-based dataset, it includes reviews and the model has to understand if it is a good one or not.
