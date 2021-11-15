import matplotlib.pyplot as plt
import numpy as np
import torch

from config.config import device


class PositionEmbedding:
    '''
     ## 位置编码的实现其实很简单，直接对照着公式去敲代码就可以，下面这个代码只是其中一种实现方式；
        ## 从理解来讲，需要注意的就是偶数和奇数在公式上有一个共同部分
        ## pos代表的是单词在句子中的索引，这点需要注意；比如max_len是128个，那么索引就是从0，1，2，...,127
        ##假设dim_model是512，2i那个符号中i从0取到了255，那么2i对应取值就是0,2,4...510
    '''
    def __init__(self, max_len=100, dim_model=1024):
        self.max_len = max_len
        self.dim_model = dim_model

    def get_angles(self, position, i, dim_model):
        # position:词语在句子中的位置[sentence_len,1]
        # i:词语在句子中的位置，[1,dim_model]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(dim_model))
        # angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / np.float32(dim_model))
        return position * angle_rates

    def get_position_embedding(self):
        angles_rads = self.get_angles(np.arange(self.max_len)[:, np.newaxis], np.arange(self.dim_model)[np.newaxis, :],
                                      self.dim_model)
        sines = np.sin(angles_rads[:, 0::2])
        # [:, 0::2]这个用法，就是从0开始到最后面，补长为2，其实代表的就是偶数位置
        # [sentence_len,dim_model]/2
        coses = np.cos(angles_rads[:, 1::2])
        # [:, 1::2]这个用法，就是从1开始到最后面，补长为2，其实代表的就是奇数位置
        # [sentence_len,dim_model]/2

        position_embedding = np.concatenate([sines, coses], axis=-1)
        position_embedding = position_embedding[np.newaxis, ...]
        position_embedding = torch.tensor(position_embedding, dtype=torch.float32, device=device)
        return position_embedding

    def plot_position_embedding(self, position_embedding, x_max):
        plt.pcolormesh(position_embedding[0], cmap="RdBu")
        plt.xlabel("Depth")
        plt.xlim(0, x_max)
        plt.ylabel("Position")
        plt.colorbar()
        plt.show()


if __name__ == '__main__':
    pe = PositionEmbedding(dim_model=512, max_len=50)
    print(pe.get_position_embedding().shape)
    # torch.Size([1, 50，512])
    pe.plot_position_embedding(pe.get_position_embedding(), x_max=512)
