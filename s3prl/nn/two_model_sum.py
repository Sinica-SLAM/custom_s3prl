import torch
import math


class two_model_sum(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        l:weight
        """
        super().__init__()
        self.l = torch.nn.Parameter(torch.zeros(()))

    def forward(self, feature_1, feature_2):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        # result = []
        # print(torch.sigmoid(self.l))
        # exit()
        l = torch.sigmoid(self.l)
        result = l * feature_1 + (1-l) * feature_2
        return result
