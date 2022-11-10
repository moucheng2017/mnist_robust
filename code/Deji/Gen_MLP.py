class GenMLP(nn.Module):
  def __init__(self, hidden, output, num_exp):
    self.hidden = hidden 
    self.num_exp = num_exp 
    self.output = output 

    experts = []
    for exp in range(self.num_exp): 
      experts.append(nn.Sequential(nn.Linear(self.hidden, self.output), 
                                   nn.BatchNorm1d(self.output), nn.ReLU())) 
    self.experts = experts 

  def forward(self, x):
    outputs = []
    for expert in self.experts:
      outputs.append(expert(x))  
      
    outputs = torch.sum(outputs) 

    return outputs 
