import torch
import torch.nn as nn
from torch import Tensor

from config.config import device
from model.feed_forward import FeedForwardNetwork
from model.multi_head_attention import MultiHeadAttention
from model.position_emdedding import PositionEmbedding


class EncoderLayer(nn.Module):
    '''
    EncoderLayer ：包含两个部分，多头注意力机制和前馈神经网络
    '''
    def __init__(self, dim_model: int, head_nums: int, dff: int, rate: float = 0.1):
        super(EncoderLayer, self).__init__()
        self.dim_model = dim_model
        self.head_nums = head_nums
        self.dff = dff
        self.enc_self_attn = MultiHeadAttention(dim_model=self.dim_model, head_nums=self.head_nums)
        self.ffn = FeedForwardNetwork(dim_model=self.dim_model, dff=self.dff)
        self.ln1 = nn.LayerNorm(self.dim_model, eps=1e-6)
        self.ln2 = nn.LayerNorm(self.dim_model, eps=1e-6)
        self.dropout1 = nn.Dropout(p=rate)
        self.dropout2 = nn.Dropout(p=rate)

    def forward(self, batch_size: int, enc_inputs: Tensor, enc_self_attn_mask: Tensor = None):
        attn_outputs, enc_attn_weights = self.enc_self_attn(batch_size=batch_size, q=enc_inputs, k=enc_inputs,
                                                            v=enc_inputs, mask=enc_self_attn_mask)
        # print("enc_self_attn")
        attn_outputs = self.dropout1(attn_outputs)
        # print(attn_outputs.shape)
        ln1_outputs = self.ln1(enc_inputs + attn_outputs)
        # 残差连接
        # (batch_size,seq_len_dim_model)
        ffn_outputs = self.ffn(ln1_outputs)
        ffn_outputs = self.dropout2(ffn_outputs)
        # (batch_size,seq_len_dim_model)
        enc_outputs = self.ln2(ln1_outputs + ffn_outputs)
        # (batch_size,seq_len_dim_model)
        # 残差连接
        # print("enc_outputs")
        return enc_outputs, enc_attn_weights


class DecoderLayer(nn.Module):
    def __init__(self, dim_model: int, head_nums: int, dff: int, rate: float = 0.1):
        super(DecoderLayer, self).__init__()
        self.dim_model = dim_model
        self.head_nums = head_nums
        self.dff = dff
        self.dec_self_attn = MultiHeadAttention(dim_model=self.dim_model, head_nums=self.head_nums)
        self.dec_enc_attn = MultiHeadAttention(dim_model=self.dim_model, head_nums=self.head_nums)
        self.ffn = FeedForwardNetwork(dim_model=self.dim_model, dff=self.dff)
        self.ffn = FeedForwardNetwork(dim_model=self.dim_model, dff=self.dff)
        self.ln1 = nn.LayerNorm(self.dim_model, eps=1e-6)
        self.ln2 = nn.LayerNorm(self.dim_model, eps=1e-6)
        self.ln3 = nn.LayerNorm(self.dim_model, eps=1e-6)
        self.dropout1 = nn.Dropout(p=rate)
        self.dropout2 = nn.Dropout(p=rate)
        self.dropout3 = nn.Dropout(p=rate)

    def forward(self, batch_size: int, dec_inputs: Tensor, enc_outputs: Tensor, dec_self_attn_mask: Tensor = None,
                dec_enc_attn_mask: Tensor = None) -> (Tensor, Tensor, Tensor):
        dec_attn_outputs1, dec_attn_weights = self.dec_self_attn(batch_size=batch_size, q=dec_inputs, k=dec_inputs,
                                                                 v=dec_inputs, mask=dec_self_attn_mask)
        dec_attn_outputs1 = self.dropout1(dec_attn_outputs1)
        ln1_outputs = self.ln1(dec_inputs + dec_attn_outputs1)
        dec_attn_outputs2, dec_enc_attn_weights = self.dec_self_attn(batch_size=batch_size, q=ln1_outputs,
                                                                     k=enc_outputs, v=enc_outputs,
                                                                     mask=dec_enc_attn_mask)
        dec_attn_outputs2 = self.dropout2(dec_attn_outputs2)
        ln2_outputs = self.ln2(dec_attn_outputs1 + dec_attn_outputs2)
        ffn_outputs = self.ffn(ln2_outputs)
        ffn_outputs = self.dropout3(ffn_outputs)
        dec_outputs = self.ln3(ln2_outputs + ffn_outputs)
        return dec_outputs, dec_attn_weights, dec_enc_attn_weights


class Encoder(nn.Module):
    '''
    Encoder 部分包含三个部分：词向量embedding，位置编码部分，注意力层及后续的前馈神经网络
    '''
    def __init__(self, layer_nums: int, input_vocab_size: int, max_len: int, dim_model: int, head_nums: int,
                 dff: int, rate: float = 0.1):
        super(Encoder, self).__init__()
        self.dim_model = dim_model
        self.layer_nums = layer_nums
        self.max_len = max_len
        self.embedding = nn.Embedding(num_embeddings=input_vocab_size, embedding_dim=self.dim_model)
        self.position_embedding = PositionEmbedding(max_len=max_len, dim_model=self.dim_model).get_position_embedding()
        self.dropout = nn.Dropout(p=rate)
        self.en_layers = [EncoderLayer(dim_model=self.dim_model, head_nums=head_nums, dff=dff, rate=rate).to(device) for _ in
                          range(layer_nums)]

    def forward(self, batch_size: int, inputs: Tensor, enc_padding_mask: Tensor = None) -> Tensor:
        # inputs:(batch_size,input_seq_len)
        input_seq_len = inputs.shape[1]
        assert input_seq_len <= self.max_len
        # inputs:(batch_size,input_seq_len,dim_model)
        inputs = self.embedding(inputs)
        inputs *= torch.sqrt(torch.tensor(self.dim_model, dtype=torch.float32))
        # inputs:(batch_size,input_seq_len,dim_model)
        # inputs的input_seq_len可能比max_len小,只切取 input_seq_len个
        # position_embedding:(1,max_len,dim_model),与inputs维度不对应，会自动广播
        inputs += self.position_embedding[:, :input_seq_len, :]
        inputs = self.dropout(inputs)
        for i in range(self.layer_nums):
            inputs, _ = self.en_layers[i](batch_size=batch_size, enc_inputs=inputs, enc_self_attn_mask=enc_padding_mask)
        # inputs:(batch_size,input_seq_len,dim_model)
        # shape不会发生变化
        return inputs


class Decoder(nn.Module):
    def __init__(self, layer_nums: int, target_vocab_size: int, max_len: int, dim_model: int, head_nums: int,
                 dff: int, rate: float = 0.1):
        super(Decoder, self).__init__()
        self.dim_model = dim_model
        self.layer_nums = layer_nums
        self.max_len = max_len
        self.embedding = nn.Embedding(num_embeddings=target_vocab_size, embedding_dim=self.dim_model)
        self.position_embedding = PositionEmbedding(max_len=max_len, dim_model=self.dim_model).get_position_embedding()
        self.dropout = nn.Dropout(p=rate)
        self.de_layers = [DecoderLayer(dim_model=self.dim_model, head_nums=head_nums, dff=dff, rate=rate).to(device) for _ in
                          range(layer_nums)]

    def forward(self, batch_size: int, inputs: Tensor, enc_outputs: Tensor, dec_mask: Tensor = None,
                enc_dec_padding_mask: Tensor = None) -> (Tensor, dict):
        # inputs:(batch_size,output_seq_len)
        output_seq_len = inputs.shape[1]
        assert output_seq_len <= self.max_len
        # inputs:(batch_size,output_seq_len,dim_model)
        inputs = self.embedding(inputs)
        inputs *= torch.sqrt(torch.tensor(self.dim_model, dtype=torch.float32))

        # inputs:(batch_size,output_seq_len,dim_model)
        # inputs的output_seq_lenn可能比max_len小,只切取 output_seq_len个
        inputs += self.position_embedding[:, :output_seq_len, :]
        inputs = self.dropout(inputs)
        attn_weights = {}
        for i in range(self.layer_nums):
            inputs, attn1, attn2 = self.de_layers[i](batch_size=batch_size,
                                                     dec_inputs=inputs,
                                                     enc_outputs=enc_outputs,
                                                     dec_self_attn_mask=dec_mask,
                                                     dec_enc_attn_mask=enc_dec_padding_mask)
            attn_weights['dec_layer{}_self_attn'.format(i + 1)] = attn1
            attn_weights['dec_layer{}_dec_enc_attn'.format(i + 1)] = attn2
        # inputs:(batch_size,output_seq_len,dim_model)
        return inputs, attn_weights


class Transformer(nn.Module):
    def __init__(self, layer_nums: int, input_vocab_size: int, target_vocab_size: int, max_len: int, dim_model: int,
                 head_nums: int, dff: int, rate: float = 0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(layer_nums=layer_nums, input_vocab_size=input_vocab_size, max_len=max_len,
                               dim_model=dim_model, head_nums=head_nums, dff=dff, rate=rate).to(device)
        self.decoder = Decoder(layer_nums=layer_nums, target_vocab_size=target_vocab_size, max_len=max_len,
                               dim_model=dim_model, head_nums=head_nums, dff=dff, rate=rate).to(device)
        self.final_layer = nn.Linear(in_features=dim_model, out_features=target_vocab_size).to(device)
        # 映射到词表空间

    def forward(self, batch_size: int, inputs: Tensor, target: Tensor, enc_padding_mask: Tensor = None,
                dec_mask: Tensor = None, enc_dec_padding_mask: Tensor = None) -> (Tensor, dict):
        # 这里有两个数据进行输入，一个是enc_inputs 形状为(batch_size, src_len)，主要是作为编码段的输入，
        # 一个dec_inputs，形状为(batch_size, tgt_len)，主要是作为解码端的输入
        # enc_inputs作为输入 形状为(batch_size, src_len)，输出由自己的函数内部指定，想要什么指定输出什么，可以是全部tokens的输出，可以是特定每一层的输出；也可以是中间某些参数的输出；
        # enc_outputs就是encoder主要的输出，
        # enc_self_attns这里没记错的是QK转置相乘之后softmax之后的矩阵值，代表的是每个单词和其他单词相关性；
        # dec_outputs 是decoder主要输出，用于后续的linear映射；
        # dec_self_attns类比于enc_self_attns 是查看每个单词对decoder中输入的其余单词的相关性；
        # dec_enc_attns是decoder中每个单词对encoder中每个单词的相关性；
        enc_outputs = self.encoder(batch_size=batch_size, inputs=inputs, enc_padding_mask=enc_padding_mask)
        dec_outputs, dec_attn_weights = self.decoder(batch_size=batch_size, inputs=target, enc_outputs=enc_outputs,
                                                     dec_mask=dec_mask, enc_dec_padding_mask=enc_dec_padding_mask)
        predictions = self.final_layer(dec_outputs)
        return predictions, dec_attn_weights


if __name__ == '__main__':
    dim_model = 512
    head_nums = 8
    dff = 2048
    batch_size = 64
    seq_len = 50
    # 测试encoder layer
    temp_encl_input = torch.nn.init.uniform_(torch.Tensor(batch_size, seq_len, dim_model))
    temp_en_layer = EncoderLayer(dim_model=dim_model, head_nums=head_nums, dff=dff)
    temp_encl_output, _ = temp_en_layer(batch_size=batch_size, enc_inputs=temp_encl_input)
    print(temp_encl_output.shape)
    # torch.Size([64, 50, 512])

    # 测试decoder layer
    seq_len = 60
    temp_decl_input = torch.nn.init.uniform_(torch.Tensor(batch_size, seq_len, dim_model))
    temp_decl_layer = DecoderLayer(dim_model=dim_model, head_nums=head_nums, dff=dff)
    temp_decl_output, attn1, attn2 = temp_decl_layer(batch_size=batch_size, dec_inputs=temp_decl_input,
                                                     enc_outputs=temp_encl_output)
    print(temp_decl_output.shape)
    # torch.Size([64, 60, 512])
    print(attn1.shape)
    # torch.Size([64, 8, 60, 60])
    print(attn2.shape)
    # torch.Size([64, 8, 60, 50])

    # 测试encoder
    layer_nums = 2
    input_vocab_size = 8500
    max_len = 1500

    seq_len = 37
    temp_en_input = torch.nn.init.uniform_(torch.Tensor(batch_size, seq_len)).long()
    temp_en = Encoder(layer_nums=layer_nums, input_vocab_size=input_vocab_size, max_len=max_len, dim_model=dim_model,
                      head_nums=head_nums, dff=dff)
    temp_en_out = temp_en(batch_size=batch_size, inputs=temp_en_input)
    print(temp_en_out.shape)
    torch.Size([64, 37, 512])

    # 测试encoder

    target_vocab_size = 8000
    seq_len = 35
    temp_de_input = torch.nn.init.uniform_(torch.Tensor(batch_size, seq_len)).long()
    # torch.Size([64, 35, 512])
    temp_de = Decoder(layer_nums=layer_nums, target_vocab_size=target_vocab_size, max_len=max_len,
                      dim_model=dim_model,
                      head_nums=head_nums, dff=dff)
    temp_de_out, temp_de_attn = temp_de(batch_size=batch_size, inputs=temp_de_input, enc_outputs=temp_en_out)
    print(temp_de_out.shape)
    # torch.Size([64, 35, 512])
    # print(len(temp_de_attn))
    # len=4,4个权重
    for key in temp_de_attn:
        print(key)
        print(temp_de_attn[key].shape)

    trm = Transformer(layer_nums=layer_nums, input_vocab_size=input_vocab_size, target_vocab_size=target_vocab_size,
                      max_len=max_len, dim_model=dim_model, head_nums=head_nums, dff=dff)
    temp_trm_input = torch.nn.init.uniform_(torch.Tensor(batch_size, 26)).long()
    temp_trm_tatget = torch.nn.init.uniform_(torch.Tensor(batch_size, 31)).long()
    temp_trm_out, temp_trm_attn = trm(batch_size=batch_size, inputs=temp_trm_input, target=temp_trm_tatget)
    print(temp_trm_out.shape)
    for key in temp_trm_attn:
        print(key, temp_trm_attn[key].shape)
    # torch.Size([64, 31, 8000])
    # dec_layer1_self_attn
    # torch.Size([64, 8, 31, 31])
    # dec_layer1_dec_enc_attn
    # torch.Size([64, 8, 31, 26])
    # dec_layer2_self_attn
    # torch.Size([64, 8, 31, 31])
    # dec_layer2_dec_enc_attn
    # torch.Size([64, 8, 31, 26])
