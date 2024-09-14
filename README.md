# Continuous CNN Implementation in PyTorch

This repository contains a PyTorch implementation of the Continuous Convolutional Neural Network (CCNN) as described in the paper [*Towards a General Purpose CNN for Long Range Dependencies in ND*](https://arxiv.org/abs/2206.03398) by Romero et al. This CCNN is capable of handling arbitrary input resolutions, dimensionalities, and data lengths using continuous convolutional kernels.

## Features

- Handles data of arbitrary resolution, dimensionality, and length without the need for task-specific architectures.
- Models long-range dependencies efficiently at every layer using continuous kernels.
- Supports both sequential (1D) and visual (2D) data.
- Works with irregularly-sampled data and test-time resolution changes.

## Repository structure
```
NN-project
├── ckconv/                 # main components
│   ├── sepflexconv.py      # SepFexConv 
│   ├── ck/                 # Contains MAGNet and 1x1 conv layers
│   ├── conv/               # Standard convolution and FFTConv
├── datamodules/            # Dataset loading and preprocessing  
├── models/                 
│   ├── ccnn.py             # Main architecture
│   ├── modules/            
        ├── s4_block.py     # S4 blocks
        ├── utils/          # Avg pool, Batch norm., dropout layer
├── config/                 # config.yaml with configuration parameters
├── main.py                 # Entry point for training and evaluation
├── requirements.txt        # Python dependencies
├── notebooks/              # notebook file for explanation
└── README.md               # Project documentation

```
   
## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/GabrieleLerani/NN-project.git
   cd NN-project
   ```
2. **Set Up the Python Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
4. **Install the Dependencies**
   ```
   pip install -r requirements.txt
   ```
## Configuration file
   The `config.yaml` file contains the configuration settings for building, training, and testing the neural network model. The file is split into different sections to handle specific parts of the model.

- **net:** Defines the architecture of the neural network, including the number of layers, channels, and the type of block used.
- **kernel:** Configures kernel layers, sizes, and convolution types.
- **load_model:** Specifies if a pre-trained model should be loaded and from where.
- **train:** Contains all parameters related to the training process (e.g., number of epochs, learning rate, optimizer).
- **test:** Contains parameters for testing the model after training.
- **data:** Provides details about the dataset being used and where it is located.
- **hydra:** Handles the logging and directory management for Hydra, the configuration manager.

### Parameters Description

#### **net:**
- `data_dim`: Dimensionality of the input data.
- `in_channels`: Number of input channels.
- `out_channels`: Number of output channels (typically for classification tasks).
- `hidden_channels`: Number of hidden channels in the layers.
- `no_blocks`: Number of blocks in the model.
- `bias`: Whether to include bias terms in the layers.
- `block_type`: Type of block used (e.g., `s4` for a specific variant).

#### **kernel:**
- `kernel_no_layers`: Number of layers in the kernel.
- `kernel_hidden_channels`: Number of hidden channels in the kernel layers.
- `kernel_size`: Size of the kernel in the convolution operation.
- `conv_type`: Type of convolution (`conv` for standard convolution).
- `fft_threshold`: Threshold for Fast Fourier Transform (used in some layers).
- `omega_0`: A parameter for initialization in some kernel layers.

#### **load_model:**
- `pre_trained`: Whether to load a pre-trained model.
- `model`: Which model to load (e.g., "last" refers to the last saved model).

#### **train:**
- `logger`: Enable or disable logging during training.
- `callbacks`: Enable or disable callbacks (e.g., early stopping, checkpoints).
- `profiler`: Enable or disable profiling to monitor performance.
- `accelerator`: Defines which accelerator to use (`cpu`, `gpu`, etc.).
- `devices`: Specifies the number of devices to use (set to "auto" for automatic selection).
- `warmup_epochs`: Number of warmup epochs (gradual learning rate increase).
- `epochs`: Total number of epochs for training.
- `learning_rate`: Learning rate for the optimizer.
- `batch_size`: Batch size used during training.
- `start_factor`: Initial learning rate multiplier.
- `end_factor`: Final learning rate multiplier.
- `dropout_rate`: Dropout rate for regularization.
- `weight_decay`: Weight decay for L2 regularization.
- `max_epoch_no_improvement`: Early stopping after a specified number of epochs without improvement.

#### **test:**
- `epochs`: Number of epochs for testing.
- `batch_size`: Batch size used during testing.

#### **data:**
- `data_dir`: Directory where datasets are stored.
- `dataset`: Dataset being used (e.g., "sc_mfcc").
- `type`: Type of data being used.
- `reduced_dataset`: Whether to use a reduced version of the dataset.
- `light_lra`: Used only with Long Range Arena (LRA) datasets.
   
### Examples
When running the `main.py` script, the user can pass the following parameters to override their default values in the `config.yaml`. 
- `kernel_no_layers`: Number of layers in the kernel.
- `kernel_hidden_channels`: Number of hidden channels in the kernel layers.
- `kernel_size`: Size of the kernel in the convolution.
- `conv_type`: Type of convolution (e.g., `conv` for standard convolution).
- `accelerator`: Device used for computation (e.g., `cpu` or `gpu`).
- `logger`: Whether to enable logging.
- `callbacks`: Whether to enable training callbacks.
- `profiler`: Whether to enable profiling for performance monitoring.
- `dataset`: The dataset to be used (overrides the one in `config.yaml`).

All the training parameters not explicetely defined are replaced with the optimal hyperparameters suggested in the original paper:
![Screenshot from 2024-09-14 16-32-02](https://github.com/user-attachments/assets/9cf6573b-0781-40bc-ab0c-5a9ca3a0b7e8)

#### Example 1: train the model with four s4 blocks, a hidden channel size of 140, and kernel size of 33 on sequencial mnist using gpu
   ```
   python main.py data.dataset=smnist train.accelerator=gpu net.hidden_channels=140 net.no_blocks=4 kernel.kernel_size=33
   ```
#### Example 2: disable logger and callbacks, if you don't want early stopping and TensorBoard logging:
   ```
   python main.py data.dataset=cifar10 train.logger=False train.callbacks=False
   ```

## Aknowledgments
We really thank the authors of the original work
```
@article{knigge2023modelling,
  title={Modelling Long Range Dependencies in N-D: From Task-Specific to a General Purpose CNN},
  author={Knigge, David M and Romero, David W and Gu, Albert and Bekkers, Erik J and Gavves, Efstratios and Tomczak, Jakub M and Hoogendoorn, Mark and Sonke, Jan-Jakob},
  journal={International Conference on Learning Representations},
  year={2023}
}
```
