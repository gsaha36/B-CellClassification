import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable

class SDLayer(nn.Module):
    def __init__(self, W):
        super(SDLayer, self).__init__()
        self.W = nn.Parameter(W)
        self.activation = nn.Tanh()

    def forward(self, x):
        PHI = self.W
        PHI = PHI.view(3, 3)
        PHI_INV = PHI.inverse()
        #PHI_INV = PHI_INV.view(3,3,1,1)

        mask = torch.tensor((1.0 - (x > 0.)) * 255.0).float()
        x = x + mask  # this image contains 255 wherever it had 0 initially

        I_OD = - torch.log(x / 255.0)
        #print(I_OD.shape)
        #print(PHI_INV.shape)
        #A = torch.mm(I_OD, PHI_INV)
        A = self.activation(I_OD)
        return A
