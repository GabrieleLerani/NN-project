import torch
import unittest
from sepflexconv import SepFlexConv
from conv import fftconv, conv as simple_conv
from omegaconf import OmegaConf


class TestSepFlexConv(unittest.TestCase):
    def setUp(self):
        data_dim = 2
        in_channels = 3
        out_channels = 140
        net_cfg = OmegaConf.create({
            'hidden_channels': 140,
            'bias': False
        })
        kernel_cfg = OmegaConf.create({
            'kernel_no_layers': 3,
            'kernel_hidden_channels': 64,
            'kernel_size': 33,
            'conv_type': 'conv',
            'fft_threshold': 50
        })
        self.model = SepFlexConv(data_dim, in_channels, out_channels, net_cfg, kernel_cfg)

    def test_forward_pass(self):
        # Create a sample input tensor with shape [64, 3, 32, 32]
        x = torch.randn(64, 3, 32, 32)
        
        # Forward pass
        masked_kernel = self.model.construct_masked_kernel(x)
        self.assertEqual(masked_kernel.shape, (1, 3, 33, 33), "Masked kernel shape mismatch")
        
        size = torch.tensor(masked_kernel.shape[2:])
        if self.model.conv_type == "fftconv" and torch.all(size > self.model.fft_threshold):
            out = fftconv(x=x, kernel=masked_kernel, bias=self.model.bias)
        else:
            out = simple_conv(x=x, kernel=masked_kernel, bias=self.model.bias)
        
        self.assertEqual(out.shape, (64, 3, 32, 32), "Spatial convolution output shape mismatch")
        
        # Pointwise convolution
        out = self.model.pointwise_conv(out)
        self.assertEqual(out.shape, (64, 140, 32, 32), "Pointwise convolution output shape mismatch")

if __name__ == '__main__':
    unittest.main()