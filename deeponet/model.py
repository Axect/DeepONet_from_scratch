import torch
from torch import nn

def create_net(sizes, hidden_activation):
    net = []
    for i in range(len(sizes)-1):
        net.append(nn.Linear(sizes[i], sizes[i+1]))
        if i < len(sizes)-2:
            net.append(hidden_activation)
    return nn.Sequential(*net)

class DeepONetScratch(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        
        self.branch_net = create_net([hparams["num_input"]] + [hparams["hidden_size"]]*(hparams["hidden_depth"]-1) + [hparams["num_branch"]], 
                                     hparams["hidden_activation"])
        self.trunk_net = create_net([hparams["dim_output"]] + [hparams["hidden_size"]]*(hparams["hidden_depth"]-1) + [hparams["num_branch"]],
                                    hparams["hidden_activation"])
        
        self.bias = nn.Parameter(torch.randn(1), requires_grad=True)

    def forward(self, u, y):  
        l = y.shape[1]
        branch_out = self.branch_net(u) 
        trunk_out = torch.stack([self.trunk_net(y[:, i:i+1]) for i in range(l)], dim=2)
        pred = torch.einsum("bp,bpl->bl", branch_out, trunk_out) + self.bias
        return pred