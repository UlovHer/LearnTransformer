import torch
from torch import Tensor
from config.config import device


class MaskUtils:

    @staticmethod
    def create_padding_mask(batch_data: Tensor) -> Tensor:
        '''
        padding mask
        '''
        padding_mask = batch_data.eq(0).type(torch.float32)
        # padding_mask = batch_data.eq(0).clone().type(torch.float32)
        return padding_mask.unsqueeze(dim=1).unsqueeze(dim=1)

    @staticmethod
    def create_look_ahead_mask(size) -> Tensor:
        look_ahead_mask = torch.triu(torch.ones(size, size, device=device),
                                     diagonal=1)
        # 生成一个上三角矩阵
        return look_ahead_mask

    def create_mask(self, input: Tensor, target: Tensor) -> (Tensor, Tensor, Tensor):
        enc_padding_mask = self.create_padding_mask(input)
        # print(enc_padding_mask.shape)
        # enc_dec_padding_mask = self.create_padding_mask(inputs)
        enc_dec_padding_mask = enc_padding_mask
        look_ahead_mask = self.create_look_ahead_mask(target.shape[1])
        dec_padding_mask = self.create_padding_mask(target)
        dec_mask = torch.max(dec_padding_mask, look_ahead_mask)
        # 获得总的deccoder input mask
        return enc_padding_mask, dec_mask, enc_dec_padding_mask


if __name__ == '__main__':
    padding_mask = torch.tensor([[7.0, 2, 0, 4, 1], [7, 0, 5, 0, 1], [7, 0, 5, 4, 0]], dtype=torch.float32)
    print(MaskUtils.create_padding_mask(padding_mask).shape)
    # print(padding_mask)
    look_ahead_mask = MaskUtils.create_look_ahead_mask(3)
    print(look_ahead_mask)
