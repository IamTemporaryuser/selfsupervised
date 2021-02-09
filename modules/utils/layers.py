import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
import math

__all__ = ["PLinear"]

class PLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(PLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.alpha = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        nn.init.uniform_(self.alpha, -math.sqrt(3), math.sqrt(3))

    def forward(self, input):
        weight = self.alpha * self.weight

        return F.linear(input, weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(self.in_features, self.out_features, self.bias is not None)