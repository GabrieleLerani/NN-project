# Continuous CNN Implementation in PyTorch

This repository contains a PyTorch implementation of the Continuous Convolutional Neural Network (CCNN) as described in the paper *Towards a General Purpose CNN for Long Range Dependencies in ND* by Romero et al. This CCNN is capable of handling arbitrary input resolutions, dimensionalities, and data lengths using continuous convolutional kernels.

## Features

- Handles data of arbitrary resolution, dimensionality, and length without the need for task-specific architectures.
- Models long-range dependencies efficiently at every layer using continuous kernels.
- Supports both sequential (1D) and visual (2D) data.
- Works with irregularly-sampled data and test-time resolution changes.

## Repository structure
   ```
├── continuous_cnn/
│   ├── __init__.py
│   ├── model.py            # Continuous CNN model implementation
│   ├── layers.py           # Custom continuous layers
│   └── utils.py            # Helper functions
├── data/
│   └── dataset.py          # Dataset loading and preprocessing
├── main.py                 # Entry point for training and evaluation
├── requirements.txt        # Python dependencies
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
## Example usage
   TODO add more details on parameters
   ```
   python main.py data.dataset=smnist train.accelerator=cpu net.hidden_channels=140 net.no_blocks=4
   ```

## Aknowledgments
We really thanks the authors of the original work.
```
@article{knigge2023modelling,
  title={Modelling Long Range Dependencies in N-D: From Task-Specific to a General Purpose CNN},
  author={Knigge, David M and Romero, David W and Gu, Albert and Bekkers, Erik J and Gavves, Efstratios and Tomczak, Jakub M and Hoogendoorn, Mark and Sonke, Jan-Jakob},
  journal={International Conference on Learning Representations},
  year={2023}
}
```
