import torch
import torch.nn as nn
import torch.nn.functional as F
# from parrallel import conv2D_para


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


    def forward(self, x):
        outputs = []
        for expert in self.experts:
            expert = expert.cuda()
            outputs.append(expert(x))
        ## Outputs is a list of Sequentials
        # print('Size of 1st sequential is', outputs[0].size())
        # outputs = torch.stack(outputs, dim=0)
        # outputs = torch.sum(outputs, dim=0)
        # print('The shape of outputs2 is ', outputs2.size())

        return outputs


class Generalised_GMoE_VI(nn.Module):
    def __init__(self, width, dilation=5, num_elayers=None, device='cuda'):
        super().__init__()

        # # the mean of the gaussian which will be added to the gate
        # self.mu = mu
        # # the std of the gaussian which will be added to the gate
        # self.sigma = sigma
        # # the amount top amount of values that will be kept once
        # self.ks = ks

        self.device = device

        self.num_elayers = num_elayers

        self.softmax = nn.Softmax()

        self.dconv_down1 = conv2D_para(1, width, 1, num_experts=num_elayers[0])  # conv_block(1, width, 1)
        self.dconv_down2 = conv2D_para(width, width, 1, num_experts=num_elayers[1])
        self.dconv_down3 = conv2D_para(width, 2 * width, 1, num_experts=num_elayers[2])

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up2 = conv2D_para(2 * width + width, width, 1, num_experts=num_elayers[3])
        self.dconv_up1 = conv2D_para(width + width, width, 1, num_experts=num_elayers[4])

        self.conv_last = nn.Conv2d(width, 1, 1)

        # create a the weight element of the gate
        self.gate1 = nn.Sequential(nn.Linear(784, num_elayers[0]))
        # create the weight element of the noise
        self.noise1 = nn.Sequential(nn.Linear(784, num_elayers[0]))

        # the following is the same as above except for the different gates
        self.gate2 = nn.Sequential(nn.Linear(1960, num_elayers[1]))
        self.noise2 = nn.Sequential(nn.Linear(1960, num_elayers[1]))

        self.gate3 = nn.Sequential(nn.Linear(490, num_elayers[2]))
        self.noise3 = nn.Sequential(nn.Linear(490, num_elayers[2]))

        self.gate4 = nn.Sequential(nn.Linear(5880, num_elayers[3]))
        self.noise4 = nn.Sequential(nn.Linear(5880, num_elayers[3]))

        self.gate5 = nn.Sequential(nn.Linear(15680, num_elayers[4]))
        self.noise5 = nn.Sequential(nn.Linear(15680, num_elayers[4]))

    def gumbel_softmax_sample(logits, temperature, eps=1e-20):
        U = torch.rand(logits).cuda()
        noise = -torch.log(-torch.log(U + eps) + eps)
        y = logits + noise
        return F.softmax(y / temperature, dim=-1)

    def eval_gate(self, x, moe_layer, gate, temperature):
        output = moe_layer(x)
        weight = gate(x)
        weight = self.gumbel_softmax_sample(weight, temperature)
        output = [output[i]*weight[:, i] for i in range(len(output))]
        output = torch.stack(output, dim=0)
        output = torch.sum(output, dim=0)
        return output, weight

    def forward(self, x, temp):
        """
        --- forward ---

        Forward pass through the network

        """
        # mix the experts
        conv1, w1  = self.eval_gate(x, self.dconv_down1(x), self.gate1, 2.0)
        x = self.maxpool(conv1)

        # mix the experts
        conv2, w2 = self.eval_gate(x, self.dconv_down2, self.gate2, 1.0)
        x = self.maxpool(conv2)

        # you guessed it we're mixing the experts again
        conv3, w3 = self.eval_gate(x, self.dconv_down3, self.gate3, 1.0)
        x = self.upsample(conv3)

        # spice it up and at a skip connection
        x = torch.cat([x, conv2], dim=1)
        # mix some experts
        x, w4 = self.eval_gate(x, self.dconv_up2, self.gate4, 0.5)
        x = self.upsample(x)

        # skip connection
        x = torch.cat([x, conv1], dim=1)
        # mix those bad bois
        x, w5 = self.eval_gate(x, self.dconv_up1, self.gate5, 0.5)
        out = self.conv_last(x)

        w = (w1+w2+w3+w4+w5) / 5
        return out, torch.mean(w, 0)