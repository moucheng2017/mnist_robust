import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from parallel_classes import conv2D_para
import numpy as np

class Generalised_GMoE(nn.Module):
    def __init__(self, width, ks, mu=0, sigma= 1, dilation=5, num_elayers = None, standard_gate = True, random_gate=None, avg_gate=None, device='cuda'):
        super().__init__()

        #the mean of the gaussian which will be added to the gate 
        self.mu = mu 
        #the std of the gaussian which will be added to the gate 
        self.sigma = sigma 
        #the amount top amount of values that will be kept once 
        self.ks = ks

        self.device = device 

        self.num_elayers = num_elayers 

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax() 

        #whether or not we will use the gate parameterised by a nn 
        self.standard_gate = standard_gate
        #whether or not we use a random gate 
        self.random_gate = random_gate 
        #whether or not we use an average gate 
        self.avg_gate = avg_gate 

        self.dconv_down1 = conv2D_para(1, width, 1, num_experts=num_elayers[0])  # conv_block(1, width, 1)
        self.dconv_down2 = conv2D_para(width, width, 1, num_experts=num_elayers[1])
        self.dconv_down3 = conv2D_para(width, 2*width, 1, num_experts=num_elayers[2])

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up2 = conv2D_para(2*width + width, width, 1, num_experts=num_elayers[3])
        self.dconv_up1 = conv2D_para(width + width, width, 1, num_experts=num_elayers[4])

        self.conv_last = nn.Conv2d(width, 1, 1)

        #create a the weight element of the gate 
        self.gate1 = nn.Sequential(nn.Linear(784, num_elayers[0]))  
        #create the weight element of the noise 
        self.noise1 = nn.Sequential(nn.Linear(784, num_elayers[0])) 

        #the following is the same as above except for the different gates  
        self.gate2 = nn.Sequential(nn.Linear(1960, num_elayers[1])) 
        self.noise2 = nn.Sequential(nn.Linear(1960, num_elayers[1])) 

        self.gate3 = nn.Sequential(nn.Linear(490, num_elayers[2])) 
        self.noise3 = nn.Sequential(nn.Linear(490, num_elayers[2])) 

        self.gate4 = nn.Sequential(nn.Linear(5880, num_elayers[3])) 
        self.noise4 = nn.Sequential(nn.Linear(5880, num_elayers[3])) 

        self.gate5 = nn.Sequential(nn.Linear(15680, num_elayers[4])) 
        self.noise5 = nn.Sequential(nn.Linear(15680, num_elayers[4])) 



    def create_gate(self, x, gate_fn, noise_fn, num_experts, k=None):
      """ 
       --- create_gate ---

       A function which will create a gate for a MoE layer 

       Args:
        - x: The input of the layer 
        - gate_fn: The nn layer used to compute the initial gate values
        - noise_fn: The noise function which produces a value that modulates the Gaussian noise  
        - num_experts: The number of experts which will be gated.  
        - k: The amount top amount of expert opinions which will be kept. 

      Out:
        - gate: Gate
      """
      if self.standard_gate:
        #compute initial gate values 
        x = x.reshape(x.size(0), -1) 
        gate = gate_fn(x)
        #compute the noise modulator 
        noise = self.softplus(noise_fn(x)) 
        #compute the value of the gate
        gate = gate + (torch.empty(num_experts).normal_(mean=self.mu,std=self.sigma).to(self.device)  *  noise)                                 
        if num_experts > 2: 
          #if we can do topk (because we have more than two values) then do it lol
          gate_sum = torch.sum(gate, dim=0)
          _, indices = torch.topk(gate_sum, k)  
          null_indices = [idx for idx in range(num_experts) if idx not in indices] 
          for nIdx in null_indices:
            gate[:, nIdx] = 0 
          
        return self.softmax(gate)

      elif self.random_gate: 
        #compute the random gate 
        gate = torch.tensor(np.random.binomial(1, 0.5, (x.size(0), num_experts))).to(self.device)
        
      elif self.avg_gate:
        #compute the average gate 
        gate = (torch.ones(x.size(0), num_experts) * (1.0/num_experts)).to(self.device)
      
      return gate 

    def eval_gate(self, x, moe_layer, gate, noise, num_experts, k):
        """
        --- eval_gate --- 

        A function which instantiates a gate and applies it to experts accordingly. 

        Args:
          - x: The input to this mixture layer. 
          - moe_layer: The MoE layer.
          - gate: The gate. 
          - noise: The noise function. 
          - k: Number of top experts to keep. 

        Output:
          - outputs: The result of the modulated mixture of experts. 
        """
        #instantiate a gate 
        gate = self.create_gate(x, gate, noise, num_experts, k)

        batch_size = x.size(0) 
        
        #compute the initial output without gating 
        init_output = moe_layer(x) 
        # print('NUM EXPERTS IS ', num_experts)
        # print('BATCH SIZE IS', batch_size)
        # print(f'Gate shape {gate.shape}')
        # print(f'tensor shape {init_output.shape}')
        for i in range(num_experts):
          for j in range(batch_size):
            init_output[i, j] = torch.mul(init_output[i, j], torch.full((init_output[i, j].shape), float(gate[j, i])).to(self.device))   
       
        #this controls whether or not we average the results following a random gate or not 
        mod = (1/num_experts) if self.random_gate else 1 

        #compute the final output 
        outputs = torch.sum(init_output, dim=0) * mod 
    
        return outputs  
           
    def forward(self, x):
        """ 
        --- forward ---

        Forward pass through the network 
        
        """

        #mix the experts 
        conv1 = self.eval_gate(x, self.dconv_down1, self.gate1, self.noise1, num_experts=self.num_elayers[0], k=self.ks[0])
        x = self.maxpool(conv1) 

        #mix the experts 
        conv2 = self.eval_gate(x, self.dconv_down2, self.gate2, self.noise2, num_experts=self.num_elayers[1], k=self.ks[1])
        x = self.maxpool(conv2)
 
        #you guessed it we're mixing the experts again 
        conv3 = self.eval_gate(x, self.dconv_down3, self.gate3, self.noise3, num_experts=self.num_elayers[2], k=self.ks[2] )
        x = self.upsample(conv3)

        #spice it up and at a skip connection 
        x = torch.cat([x, conv2], dim=1) 
        #mix some experts 
        x = self.eval_gate(x, self.dconv_up2, self.gate4, self.noise4, num_experts=self.num_elayers[3], k=self.ks[3])
        x = self.upsample(x) 

        #skip connection
        x = torch.cat([x, conv1], dim=1)
        #mix those bad bois 
        x = self.eval_gate(x, self.dconv_up1, self.gate5, self.noise5, num_experts=self.num_elayers[4], k=self.ks[4])
        out = self.conv_last(x)

        return out 