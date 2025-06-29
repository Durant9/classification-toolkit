import torch.nn as nn
import torch

class CNNClassifier(nn.Module):
    def __init__(self, config):
        assert len(config['down_ch']) == len(config['layers']) + 1, "Number of down cannels must be n_layers + 1"
        self.down_ch = config['down_ch']
        self.layers = config['layers']
        self.im_ch = config['im_ch']
        self.num_classes = config['num_classes']
        self.im_size = config['im_size']
        self.head_features = config['head_features']
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.BatchNorm2d(self.im_ch),
            nn.Conv2d(in_channels=self.im_ch, out_channels=self.down_ch[0], kernel_size=3, padding=1, stride=1),
            nn.ReLU())

        self.convs = nn.ModuleList([])
        for i in range(len(self.layers)):
            block = [nn.BatchNorm2d(self.down_ch[i])]
            for j in range(self.layers[i]):
                block.append(nn.Conv2d(in_channels=self.down_ch[i] if j==0 else self.down_ch[i+1], out_channels=self.down_ch[i+1],
                                       kernel_size=3, padding=1, stride=1))
                block.append(nn.ReLU())
            block.append(nn.MaxPool2d(kernel_size=2))
            self.convs.append(nn.Sequential(*block))

        self.head = nn.Sequential(
            nn.Linear(in_features=self.down_ch[-1]*(self.im_size // (2**len(self.layers)))**2, out_features=self.head_features[0]),
            nn.ReLU()
        )
        for i in range(len(self.head_features)-1):
            self.head.append(nn.Linear(in_features=self.head_features[i], out_features=self.head_features[i+1]))
            self.head.append(nn.ReLU())
        self.head.append(nn.Linear(in_features=self.head_features[-1], out_features=self.num_classes))
        
    def forward(self, x):
        out = self.input_conv(x)
        for i in range(len(self.convs)):
            out = self.convs[i](out)
        out = torch.flatten(out, start_dim=1)
        out = self.head(out)
        return out