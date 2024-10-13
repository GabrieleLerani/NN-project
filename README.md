# Continuous CNN Implementation in PyTorch

This repository contains a PyTorch implementation of the Continuous Convolutional Neural Network (CCNN) as described in the paper [*Towards a General Purpose CNN for Long Range Dependencies in ND*](https://arxiv.org/abs/2206.03398) by Romero et al. This CCNN is capable of handling arbitrary input resolutions, dimensionalities, and data lengths using continuous convolutional kernels.

## :sparkles: Features

- Handles data of arbitrary resolution, dimensionality, and length without the need for task-specific architectures.
- Models long-range dependencies efficiently at every layer using continuous kernels.
- Supports both sequential (1D) and visual (2D) data.
- Works with irregularly-sampled data.

## :file_folder: Repository structure
```
NN-project
├── ckconv/                 # main components
│   ├── sepflexconv.py      # SepFexConv 
│   ├── ck/                 # Contains MAGNet and 1x1 conv layers
│   ├── conv/               # Standard convolution and FFTConv
├── datamodules/            # Dataset loading and preprocessing  
├── models/                 
    ├── ccnn.py             # Main architecture
    ├── modules/            
        ├── s4_block.py     # S4 blocks
        ├── utils/          # Avg pool, Batch norm., dropout layer
├── config/                 # config.yaml with configuration parameters
├── checkpoints/            # contains checkpoints of the models
├── main.py                 # Entry point for training and evaluation
├── requirements.txt        # Python dependencies
├── notebooks/              # notebook file for explanation
└── README.md               # Project documentation

```
## :computer: Technologies Used

- <img src="https://github.com/user-attachments/assets/378e3efa-969d-43be-a9ef-e9e216b706e0" alt="PyTorch Logo" width="20"> **PyTorch**: used for defining the neural network model and performing tensor operations.
- <img src="https://github.com/user-attachments/assets/953f0cdb-0047-4ae1-8a15-920e3f17b269" alt="Lightning Logo" width="20">**PyTorch Lightning**: Used to reduce the boilerplate code required for training and to structure the training loop in a cleaner, more manageable way.
  
- <img src="https://github.com/user-attachments/assets/0c30078f-adab-4a04-bf91-0011e3fa6737" alt="Tensorboard Logo" width="20"> **Tensorborad**: Used to easily visualize and track metrics improvement during training.
- <img src="https://github.com/user-attachments/assets/f1865c35-13c1-4b22-b6d7-c8ec75e313da" alt="Hydra Logo" width="20"> **Hydra**: The [`config.yaml`](config/config.yaml) file is managed using OmegaConf, which allows for flexible and hierarchical configuration management.
  
## :orange_book: Note on training and dataset
Since our limited hardware resources we trained the model using the VM of **Google Colab** with a **T4 GPU, 12 GB RAM, 100 GB on disk**. It allowed us to test on 1D and 2D dataset but it has been not enough for multi dimensional data which requires more power and memory.

We are using a **light version of the Long Range Arena (LRA)** dataset due to memory constraints. The full LRA dataset requires a substantial amount of memory, which can make it difficult to run on machines with limited resources. As a solution, we include only two tasks from the LRA dataset:

- **Pathfinder32**: The baseline version of the Pathfinder task, which involves understanding if there is a path connecting two points.
- **ListOps1000**: A task that involves learning hierarchical structures and operations on sequences up to a length of 1000.


## :wrench: Installation

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
## :clipboard: Configuration file
   The [`config.yaml`](config/config.yaml) file contains the configuration settings for building, training, and testing the neural network model. The file is split into different sections to handle specific parts of the model.

- **net:** Defines the architecture of the neural network, including the number of layers, channels, and the type of block used.
- **kernel:** Configures kernel layers, sizes, and convolution types.
- **load_model:** Specifies if a pre-trained model should be loaded and from where.
- **train:** Contains all parameters related to the training process (e.g., number of epochs, learning rate, optimizer).
- **test:** Contains parameters for testing the model after training.
- **data:** Provides details about the dataset being used and where it is located.
- **hydra:** Handles the logging and directory management for Hydra, the configuration manager.

Refer to [CONFIG.md](config/CONFIG.md) for a complete description of the parameters.
   
### Examples
When running the `main.py` script, the user can pass the following parameters to override their default values in the [`config.yaml`](config/config.yaml). 
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

During training, checkpoints are automatically saved into `checkpoints/` (created if it does not already exist) indeed pytorch_lightning allows to save the last or the best one.

#### Example 1: train the model with four s4 blocks, 140 hidden channels, and kernel size of 33 on sequential mnist using gpu:
   ```
   python main.py data.dataset=s_mnist train.accelerator=cpu net.hidden_channels=140 net.no_blocks=4 kernel.kernel_size=33
   ```
#### Example 2: start training from the top checkpoint of SMNIST:
   ```
   python main.py data.dataset=s_mnist train.accelerator=cpu net.hidden_channels=140 net.no_blocks=4 kernel.kernel_size=33 load_model.pre_trained=True load_model.model=top
   ```
#### Example 3: disable logger and callbacks, if you don't want early stopping and TensorBoard logging:
   ```
   python main.py data.dataset=cifar10 train.logger=False train.callbacks=False
   ```

## :clap: Aknowledgments
We really thank the authors of the original work
```
@article{knigge2023modelling,
  title={Modelling Long Range Dependencies in N-D: From Task-Specific to a General Purpose CNN},
  author={Knigge, David M and Romero, David W and Gu, Albert and Bekkers, Erik J and Gavves, Efstratios and Tomczak, Jakub M and Hoogendoorn, Mark and Sonke, Jan-Jakob},
  journal={International Conference on Learning Representations},
  year={2023}
}
```
