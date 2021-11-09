import numpy as np

import torch
import torch.nn as nn
import torch.functional as F

from collections import OrderedDict

class LogisticActivation(nn.Module):
    """
    Implementation of Generalized Sigmoid
    Applies the element-wise function:
    .. math::
        \text{Sigmoid}(x) = \sigma(x) = \frac{1}{1 + \exp(-k(x-x_0))}
    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional dimensions
        - Output: :math:`(N, *)`, same shape as the input
    Parameters:
        - x0: The value of the sigmoid midpoint
        - k: The slope of the sigmoid - trainable
    Examples:
        >>> logAct = LogisticActivation(0, 5)
        >>> x = torch.randn(256)
        >>> x = logAct(x)
    """

    def __init__(self, x0 = 0, k = 1, train=False):
        """
        Initialization
        INPUT:
            - x0: The value of the sigmoid midpoint
            - k: The slope of the sigmoid - trainable
            - train: Whether to make k a trainable parameter
            x0 and k are initialized to 0,1 respectively
            Behaves the same as torch.sigmoid by default
        """
        super(LogisticActivation,self).__init__()
        self.x0 = x0
        self.k = nn.Parameter(torch.FloatTensor([float(k)]))
        self.k.requiresGrad = train

    def forward(self, x):
        """
        Applies the function to the input elementwise
        """
        o = torch.clamp(1 / (1 + torch.exp(-self.k * (x - self.x0))), min=0, max=1).squeeze()
        return o

    def clip(self):
        self.k.data.clamp_(min=0)

class IdentityActivation(nn.Module):
    def forward(self, x):
        return x

class GlobalMaxPool(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, C):
        return nn.functional.adaptive_max_pool2d(C,output_size=1).reshape(1,-1)
    
class GlobalMeanPool(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, C):
        return nn.functional.adaptive_avg_pool2d(C,output_size=1).reshape(1,-1)

class ModelInteraction(nn.Module):
    def __init__(self, embedding, contact, hidden_dim=100, activation_k = 10, p=0.5, global_pool='max'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.activation_k = activation_k
        self.p = p
        self.global_pool = global_pool

        self.embedding = embedding
        self.contact = contact

        self.globalPoolDict = nn.ModuleDict({
                                    'max': GlobalMaxPool(),
                                    'mean': GlobalMeanPool()
                                    })
        
        
        self.interaction = nn.Sequential(OrderedDict({                    
            'dense1': nn.Linear(self.contact.output_dim, self.hidden_dim),
            'dense1Activation': nn.ReLU(),
            'drop1': nn.Dropout(p = self.p),
            'dense2': nn.Linear(self.hidden_dim,1),
            'drop2': nn.Dropout(p = self.p),
            'activation': LogisticActivation(x0=0.5, k = activation_k)
        }))
                                         
        self.map = nn.Sequential(OrderedDict({
            'cMapDense': nn.Conv2d(self.contact.output_dim, 1, 1),
            'cMapActivate': nn.Sigmoid()
        }))
        
        self.clip()

    def clip(self):
        self.contact.clip()

    def embed(self, x):
        if self.embedding is None:
            return x
        else:
            return self.embedding(x)
    
    def _contact_conv(self, z0, z1):
        e0 = self.embed(z0)
        e1 = self.embed(z1)
        C = self.contact(e0, e1)
        return C
    
    def cmap(self, z0, z1):
        e0 = self.embed(z0)
        e1 = self.embed(z1)
        C = self.contact(e0, e1)
        cmap = self.map(C)
        return cmap.squeeze()
    
    def forward(self, z0, z1):
        e0 = self.embed(z0)
        e1 = self.embed(z1)
        C = self.contact(e0, e1)
        gp = self.globalPoolDict[self.global_pool](C)
        phat = self.interaction(gp)

        return phat.squeeze()

    def map_predict(self, z0, z1):
        C = self._contact_conv(z0, z1)
        cmap = self.map(C)
        
        gp = self.globalPoolDict[self.global_pool](C)
        phat = self.interaction(gp)
        phat = phat.squeeze()
        return cmap, phat
        