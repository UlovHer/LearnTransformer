import torch
from torch import Tensor

from utils.attention_utils import AttentionUtils


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, dim_model: int, head_nums: int):
        super(MultiHeadAttention, self).__init__()
        self.dim_model = dim_model
        self.head_nums = head_nums
        # self.dim_k = dim_k
        # self.dim_v = dim_v
        assert self.dim_model % self.head_nums == 0

        self.depth = self.dim_model // self.head_nums
        self.WQ = torch.nn.Linear(self.dim_model, self.dim_model, bias=False)
        self.WK = torch.nn.Linear(self.dim_model, self.dim_model, bias=False)
        self.WV = torch.nn.Linear(self.dim_model, self.dim_model, bias=False)

    def forward(self, batch_size: int, q: Tensor, k: Tensor, v: Tensor, mask: Tensor = None) -> (Tensor, Tensor):
        # 输入进来的QKV是相等的，使用映射linear做一个映射得到参数矩阵Wq, Wk,Wv
        # 首先映射分头，然后计算atten_scores，然后计算atten_value
        # (batch_size,seq_len,dim_model)
        # -Linear-> (batch_size,seq_len,dim_model)
        # -split-> (batch_size,seq_len,head_nums,depth)
        # -trans-> (batch_size,head_nums,seq_len,depth)
        q = self.WQ(q)
        # print(q.is_cuda)
        k = self.WK(k)
        v = self.WK(v)

        q = self.split(q, batch_size)
        k = self.split(k, batch_size)
        v = self.split(v, batch_size)
        # print(q.shape)
        sdpa_output, att_weights = AttentionUtils.scaled_dot_product_attention(q=q, k=k, v=v, mask=mask)
        # sdpa_output.permute(0, 2, 1, 3)
        sdpa_output = sdpa_output.transpose(1, 2)
        # print(sdpa_output.shape)
        output = sdpa_output.reshape(batch_size, -1, self.dim_model)
        # print(output.shape)
        return output, att_weights

    def split(self, x: Tensor, batch_size: int) -> Tensor:
        '''
        x.shape:(batch_size,seq_len,dim_model)
        dim_model = head_nums*depth
        x-> reshape(batch_size,head_nums,sep_len,depth)
        '''
        # print(x.shape)
        # x = torch.reshape(x, (batch_size, -1, self.head_nums, self.depth))
        # return x.permute(0, 2, 1, 3)
        x = x.view(batch_size, -1, self.head_nums, self.depth).transpose(1, 2)
        return x


if __name__ == '__main__':
    batch_size, seq_len_q, dim_model, head_nums = 16, 60, 512, 8
    temp_mha = MultiHeadAttention(dim_model=dim_model, head_nums=head_nums)
    q = torch.nn.init.uniform_(torch.Tensor(batch_size, seq_len_q, dim_model))
    print(q.shape)
    # q torch.Size([16, 60, 512])

    output, atten = temp_mha(batch_size=batch_size, q=q, k=q, v=q)
    print(output.shape)
    print(atten.shape)
