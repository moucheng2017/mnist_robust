import torch
import torch.nn as nn


## arbitrary number of parallel MLP
class MLP_para(nn.Module):
    def __init__(self, hidden, output, num_exp):
        self.hidden = hidden
        self.num_exp = num_exp
        self.output = output
        super().__init__()

        experts = []
        for exp in range(self.num_exp):
            experts.append(nn.Sequential(nn.Linear(self.hidden, self.output),
                                         nn.BatchNorm1d(self.output), nn.ReLU()))
        self.experts = experts

    def forward(self, x):
        outputs = []
        for expert in self.experts:
            expert = expert.cuda()
            outputs.append(expert(x))

        outputs = torch.stack(outputs, dim=0)
        outputs = torch.sum(outputs, dim=0)

        return outputs

## arbitrary number of parallel 2D convolution
class conv2D_para(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dilation=1,
                 num_experts: int = 2,
                 kernel_size: int = 3):
        self.num_experts = num_experts
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        super().__init__()


        experts = []
        for expert in range(self.num_experts):
            experts.append(nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, padding=dilation, stride=1,
                          bias=False, dilation=dilation),
                nn.BatchNorm2d(self.out_channels, affine=True),
                nn.ReLU(inplace=True)
            ))
        self.experts = experts


    def forward(self, x, sum=None):
        outputs = []
        for expert in self.experts:
            expert = expert.cuda()
            outputs.append(expert(x))
        ## Outputs is a list of Sequentials
        outputs = torch.stack(outputs, dim=0)
        if sum:
            outputs = torch.sum(outputs, dim=0)

        return outputs
