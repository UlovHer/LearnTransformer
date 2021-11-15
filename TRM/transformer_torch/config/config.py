import torch

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

# Encoder、Decoder个数
layer_nums: int = 6

# 多注意力抽头个数
head_nums: int = 8

# 模型维度
# scentence_len: int = 50
# 词向量维度
dim_model: int = 512
max_len = 1000
dff = 2048
# input_vocab_size = 8500
# target_vocab_size = 8000
input_vocab_size = 6
target_vocab_size = 9
# 训练配置
epochs = 1000
# batch_size = 48
batch_size = 2
checkpoint_folder = './checkpoint/'
