import numpy as np
import torch

from config.config import device


class AttentionUtils:
    @staticmethod
    def scaled_dot_product_attention(q, k, v, mask):
        '''
        缩放点积注意力
        输入进来的维度分别是
        q: (batch_size,head_nums, len_q, depth)
        K：(batch_size,head_nums, len_k, depth)
        V: (batch_size,head_nums, len_k, depth)
        '''
        # print(q.shape)
        matmul_qk = torch.matmul(q, k.transpose(-1, -2))
        #首先经过matmul函数得到的scores形状是 : (batch_size,head_nums,len_q,len_k)
        # print(matmul_qk.shape)
        # dk = torch.tensor(k.shape[-1], dtype=torch.float32)
        # scaled_attention_logits = matmul_qk / torch.sqrt(dk)
        scaled_attention_logits = matmul_qk / np.sqrt(k.shape[-1])
        # print(scaled_attention_logits.shape)

        if mask is not None:
            # print(mask.shape)
            scaled_attention_logits += (mask * -1e9)
            # 把被mask的地方置为无限小，softmax之后基本就是0，对q的单词不起作用

        attention_weights = torch.nn.functional.softmax(scaled_attention_logits, dim=-1)
        # print(attention_weights.shape)
        output = torch.matmul(attention_weights, v)
        # print(output.is_cuda)
        return output, attention_weights

    def print_scaled_dot_product_attention(self, q, k, v, mask=None):
        '''
        打印缩放点积注意力
        '''
        temp_out, temp_att = self.scaled_dot_product_attention(q, k, v, mask)
        print("Attention weights are:")
        print(temp_att)
        print("Output is:")
        print(temp_out)


if __name__ == '__main__':
    temp_q1 = torch.tensor([[0, 10, 0]], dtype=torch.float32)
    # (1,3)
    temp_k = torch.tensor([[10, 0, 0], [0, 10, 0], [0, 0, 10], [0, 0, 10]], dtype=torch.float32)
    # (4,3)转置后(3,4)
    temp_v = torch.tensor([[1, 0], [10, 0], [100, 5], [1000, 6]], dtype=torch.float32)
    # (4,2)
    att_utils = AttentionUtils()
    np.set_printoptions(suppress=True)
    att_utils.print_scaled_dot_product_attention(temp_q1, temp_k, temp_v)

    temp_q2 = torch.tensor([[0, 0, 10]], dtype=torch.float32)
    att_utils.print_scaled_dot_product_attention(temp_q2, temp_k, temp_v)

    temp_q3 = torch.tensor([[10, 10, 0]], dtype=torch.float32)
    att_utils.print_scaled_dot_product_attention(temp_q3, temp_k, temp_v)

    temp_q4 = torch.cat((temp_q1, temp_q2, temp_q3), dim=0)
    print(temp_q4)
    att_utils.print_scaled_dot_product_attention(temp_q4, temp_k, temp_v)
