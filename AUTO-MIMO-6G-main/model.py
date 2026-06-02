import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        # Identity mapping if channels change
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        out = self.dropout(out)
        return out

class AutoMIMONet(nn.Module):
    def __init__(self, nr, nt, hpo_params=None):
        super(AutoMIMONet, self).__init__()
        
        # Default params if not provided by HPO
        if hpo_params is None:
            hpo_params = {
                'num_blocks': 3,
                'hidden_dim': 64,
                'dropout': 0.1
            }
            
        self.nr = nr
        self.nt = nt
        
        # Initial Convolution: Input is [B, 2, Nr, Nt]
        # We treat the complex parts as channels
        self.start_conv = nn.Conv2d(2, hpo_params['hidden_dim'], kernel_size=3, padding=1)
        self.bn_start = nn.BatchNorm2d(hpo_params['hidden_dim'])
        
        # Residual Blocks
        layers = []
        for _ in range(hpo_params['num_blocks']):
            layers.append(ResBlock(hpo_params['hidden_dim'], hpo_params['hidden_dim'], hpo_params['dropout']))
        self.res_layers = nn.Sequential(*layers)
        
        # Final Convolution to bring back to [B, 2, Nr, Nt]
        self.end_conv = nn.Conv2d(hpo_params['hidden_dim'], 2, kernel_size=3, padding=1)

    def forward(self, x):
        # x: [Batch, 2, Nr, Nt]
        out = F.relu(self.bn_start(self.start_conv(x)))
        out = self.res_layers(out)
        out = self.end_conv(out)
        return out

if __name__ == "__main__":
    # Test model
    nr, nt = 16, 64
    model = AutoMIMONet(nr, nt)
    test_input = torch.randn(8, 2, nr, nt)
    output = model(test_input)
    print(f"Output Shape: {output.shape}") # Should be [8, 2, 16, 64]
