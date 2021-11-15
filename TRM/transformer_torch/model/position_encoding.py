import torch
from torch import nn
import numpy as np
from matplotlib import pyplot as plt

from config.config import device


class PositionEncoding(nn.Module):
    def __init__(self, max_len=100, dim_model=1024, dropout=0.1):
        super(PositionEncoding, self).__init__()
        self.max_len = max_len
        self.dim_model = dim_model
        self.dropout = nn.Dropout(p=dropout)
        position_embedding = self.get_position_embedding()
        self.register_buffer('position_embedding', position_embedding)

    def forward(self, x):
        '''
            x: [seq_len, batch_size, d_model]
        '''
        x = x + self.position_embedding[:x.size(0), :]
        return self.dropout(x)

    def get_angles(self, position, i, dim_model):
        # position:词语在句子中的位置[sentence_len,1]
        # i:词语在句子中的位置，[1,dim_model]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(dim_model))
        # angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / np.float32(dim_model))
        return position * angle_rates

    # def get_position_embedding(self, sentence_len, dim_model):
    def get_position_embedding(self):
        angles_rads = self.get_angles(np.arange(self.max_len)[:, np.newaxis], np.arange(self.dim_model)[np.newaxis, :],
                                      self.dim_model)
        sines = np.sin(angles_rads[:, 0::2])
        # [sentence_len,dim_model]/2
        coses = np.cos(angles_rads[:, 1::2])
        # [sentence_len,dim_model]/2

        position_embedding = np.concatenate([sines, coses], axis=-1)
        position_embedding = position_embedding[np.newaxis, ...]
        return torch.tensor(position_embedding, dtype=torch.float32, device=device)

    def plot_position_embedding(self, position_embedding, x_max):
        plt.pcolormesh(position_embedding[0], cmap="RdBu")
        plt.xlabel("Depth")
        plt.xlim(0, x_max)
        plt.ylabel("Position")
        plt.colorbar()
        plt.show()


if __name__ == '__main__':
    pe = PositionEncoding(dim_model=512, max_len=50)
    print(pe.position_embedding.shape)
    # torch.Size([1, 50，512])
    pe.plot_position_embedding(pe.position_embedding, x_max=512)
