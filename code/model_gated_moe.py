def conv_block(in_channels, out_channels, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=dilation, stride=1, bias=False, dilation=dilation),
        nn.BatchNorm2d(out_channels, affine=True),
        nn.ReLU(inplace=True)
    )

class GatedMoE(nn.Module):

    def __init__(self, width, parrallel_num, k, mu=0, sigma= 1, dilation=5, standard_gate = True, random_gate=None, avg_gate=None):
        super().__init__()

        #this is the number of sub-networks per layer 
        self.parrallel_num = parrallel_num 
        #the mean of the gaussian which will be added to the gate 
        self.mu = mu 
        #the std of the gaussian which will be added to the gate 
        self.sigma = sigma 
        #the amount top amount of values that will be kept once 
        self.k = k 

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax() 

        #whether or not we will use the gate parameterised by a nn 
        self.standard_gate = standard_gate
        #whether or not we use a random gate 
        self.random_gate = random_gate 
        #whether or not we use an average gate 
        self.avg_gate = avg_gate 

        self.dconv_down1a = conv_block(1, width, 1)
        self.dconv_down2a = conv_block(width, width, 1)
        self.dconv_down3a = conv_block(width, 2*width, 1)

        self.dconv_down1b = conv_block(1, width, dilation)
        self.dconv_down2b = conv_block(width, width, dilation)
        self.dconv_down3b = conv_block(width, 2*width, dilation)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up2a = conv_block(2*width + width, width, 1)
        self.dconv_up1a = conv_block(width + width, width, 1)

        self.dconv_up2b = conv_block(2*width + width, width, dilation)
        self.dconv_up1b = conv_block(width + width, width, dilation)
        self.conv_last = nn.Conv2d(width, 1, 1)
        
        #create a the weight element of the gate 
        self.gate1 = nn.Sequential(nn.Linear(784, parrallel_num))  
        #create the weight element of the noise 
        self.noise1 = nn.Sequential(nn.Linear(784, parrallel_num)) 

        #the following is the same as above except for the different gates  
        self.gate2 = nn.Sequential(nn.Linear(1960, parrallel_num)) 
        self.noise2 = nn.Sequential(nn.Linear(1960, parrallel_num)) 

        self.gate3 = nn.Sequential(nn.Linear(490, parrallel_num)) 
        self.noise3 = nn.Sequential(nn.Linear(490, parrallel_num)) 

        self.gate4 = nn.Sequential(nn.Linear(5880, parrallel_num)) 
        self.noise4 = nn.Sequential(nn.Linear(5880, parrallel_num)) 

        self.gate5 = nn.Sequential(nn.Linear(15680, parrallel_num)) 
        self.noise5 = nn.Sequential(nn.Linear(15680, parrallel_num)) 



    def create_gate(self, x, gate_fn, noise_fn):
      """ 
       --- create_gate ---

       A function which will create a gate for a MoE layer 

       Args:
        - x: The input of the layer 
        - gate_fn: The nn layer used to compute the initial gate values
        - The noise function which produces a value that modulates the Gaussian noise  
      """
      if self.standard_gate:
        #compute initial gate values 
        x = x.reshape(x.size(0), -1) 
        gate = gate_fn(x)
        #compute the noise modulator 
        noise = self.softplus(noise_fn(x)) 
        #compute the value of the gate
        gate = gate + (torch.empty(self.parrallel_num).normal_(mean=self.mu,std=self.sigma).to('cuda')  *  noise)                                 
        if self.parrallel_num > 2: 
          #if we can do topk (because we have more than two values) then do it lol
          gate, _ = torch.topk(gate, self.k) 

        return self.softmax(gate) 

      elif self.random_gate: 
        #compute the random gate 
        gate = np.random.binomial(self.parrallel_num, 0.5) 
        

      elif self.avg_gate:
        #compute the average gate 
        gate = torch.ones(self.parrallel_num) * (1.0/self.parrallel_num) 

      return gate 

    def eval_gate(self, x, conv1, conv2, gate, noise):
        """
        --- eval_gate --- 

        A function which instantiates a gate and applies it to experts accordingly. 

        Args:
          - x: The input to this mixture layer. 
          - conv1: The first expert 
          - conv2: The second expert 
          - gate: The gate function
          - noise: The noise function 

        Output:
          - out: The result of the modulated mixture of experts. 
        """
        #instantiate a gate 
        gate = self.create_gate(x, gate, noise)
        
        #compute the expert's opinion
        conv1a = conv1(x) 
        #modulate the output of the first expert by the corresponding gate index 
        conv1a = torch.mul(conv1a, torch.full((conv1a.shape), float(gate[0][0])).to('cuda'))  
        
        #compute the expert's opinion
        conv1b = conv2(x)
        #modulate the output of the second expert by the corresponding gate index 
        conv1b = torch.mul(conv1b, torch.full(conv1b.shape, float(gate[0][1])).to('cuda'))  
       
        #this controls whether or not we average the results following a random gate or not 
        mod = 0.5 if self.random_gate else 1 

        #compute the final output 
        out = (conv1a + conv1b) * mod 
        
        return out 
           
    def forward(self, x):
        """ 
        --- forward ---

        Forward pass through the network 
        
        """
        #mix the experts 
        conv1 = self.eval_gate(x, self.dconv_down1a, self.dconv_down1b, self.gate1, self.noise1)
        x = self.maxpool(conv1) 

        #mix the experts 
        conv2 = self.eval_gate(x, self.dconv_down2a, self.dconv_down2b, self.gate2, self.noise2)
        x = self.maxpool(conv2)
 
        #you guessed it we're mixing the experts again 
        conv3 = self.eval_gate(x, self.dconv_down3a, self.dconv_down3b, self.gate3, self.noise3)
        x = self.upsample(conv3)

        #spice it up and at a skip connection 
        x = torch.cat([x, conv2], dim=1) 
        #mix some experts 
        x = self.eval_gate(x, self.dconv_up2a, self.dconv_up2b, self.gate4, self.noise4)
        x = self.upsample(x) 

        #skip connection
        x = torch.cat([x, conv1], dim=1)
        #mix those bad bois 
        x = self.eval_gate(x, self.dconv_up1a, self.dconv_up1b, self.gate5, self.noise5)
        out = self.conv_last(x)

        return out

