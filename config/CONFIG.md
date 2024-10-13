## Parameters Description

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
- `dataset`: Dataset being used, refer to [DATAMODULES.md](datamodules/DATAMODULES.md) for more details.
- `type`: Type of data being used.
- `reduced_dataset`: Whether to use a reduced version of the dataset.
- `light_lra`: Used only with Long Range Arena (LRA) datasets.
