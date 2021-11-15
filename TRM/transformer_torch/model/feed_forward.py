import torch
import torch.nn as nn
from torch import Tensor

from config.config import device


class FeedForwardNetwork(nn.Module):
    def __init__(self, dim_model: int, dff: int):
        '''
        dff：dim of feed forward network
        '''
        # self.dim_model = dim_model
        super(FeedForwardNetwork, self).__init__()
        self.fc = nn.Sequential(nn.Linear(dim_model, dff, bias=False), nn.ReLU(),
                                nn.Linear(dff, dim_model, bias=False))
        # self.ln = nn.LayerNorm(self.dim_model)

    def forward(self, inputs: Tensor):
        # residual = inputs
        # print(inputs.is_cuda)
        inputs = self.fc(inputs)
        # print(inputs.is_cuda)
        # outputs=self.ln(outputs)
        # print(outputs.shape)
        return inputs


if __name__ == '__main__':
    dim_model = 512
    dff = 2048
    temp_ffn = torch.nn.init.uniform_(torch.Tensor(64, 50, 512))
    ffn = FeedForwardNetwork(dim_model=dim_model, dff=dff)
    print(ffn(temp_ffn).shape)
