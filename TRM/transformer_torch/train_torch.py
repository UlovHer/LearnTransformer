import os
import time

import torch
from torch import Tensor, optim
from torch.utils import data

from config.config import layer_nums, max_len, dim_model, head_nums, dff, epochs, batch_size, \
    input_vocab_size, target_vocab_size, device, checkpoint_folder
from model.transformer import Transformer
from utils.data_utils import make_data, DataSet
from utils.mask_utils import MaskUtils

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def loss_function(predict: Tensor, real: Tensor, loss_obj):
    mask = torch.logical_not(real.eq(0))
    # real和0比较，mask中padding处为0，
    loss = loss_obj(predict, real)
    mask = mask.type(loss.dtype)
    loss *= mask
    return torch.mean(loss)

# 将模型加载到设备中
def load_model(model: Transformer):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.__eq__("cuda"):
        # model.to(device)
        model.to(device)
    print("The model will be running on", device, "device")

# 将数据加载到设备中
def load_data(input: Tensor, tar_input: Tensor, tar_real: Tensor):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The data will be running on", device, "device")
    if device.__eq__("cuda"):
        return input.to(device), tar_input.to(device), tar_real.to(device)
    else:
        return input, tar_input, tar_real

def save_model(model: Transformer, path: str):
    print('model save path: ' + path)
    torch.save(model.state_dict(), path)


def greedy_decoder(model, enc_input, start_symbol, enc_padding_mask=None):
    """
    For simplicity, a Greedy Decoder is Beam search when K=1. This is necessary for inference as we don't know the
    target sequence input. Therefore we try to generate the target input word by word, then feed it into the transformer.
    Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
    :param model: Transformer Model
    :param enc_input: The encoder input
    :param start_symbol: The start symbol. In this example it is 'S' which corresponds to index 4
    :return: The target input
    """
    enc_outputs = model.encoder(batch_size=1, inputs=enc_input,
                                enc_padding_mask=enc_padding_mask)
    dec_input = torch.zeros(1, 1).type_as(enc_input.data).to(device)
    terminal = False
    next_symbol = start_symbol
    mu = MaskUtils()
    while not terminal:
        dec_input = torch.cat([dec_input.detach(), torch.tensor([[next_symbol]], dtype=enc_input.dtype).to(device)], -1)
        _, dec_mask, enc_dec_padding_mask = mu.create_mask(input=enc_input, target=dec_input)
        dec_outputs, _ = model.decoder(batch_size=1, inputs=dec_input, enc_outputs=enc_input,
                                       dec_mask=dec_mask,
                                       enc_dec_padding_mask=enc_dec_padding_mask)
        projected = model.final_layer(dec_outputs)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_word = prob.data[-1]
        next_symbol = next_word
        if next_symbol == tgt_vocab["."]:
            terminal = True
        print(next_word)
    return dec_input


def train(train_loader, trm_model):
    # loss_obj = torch.nn.CrossEntropyLoss(ignore_index=0, reduction="none")
    loss_obj = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(trm_model.parameters(), lr=1e-3, momentum=0.99)
    mu = MaskUtils()
    # val_loader = data_loader()
    for epoch in range(epochs):
        total_train_loss = 0.0
        total_accuracy = 0.0
        total_vall_loss = 0.0
        for input, tar_input, tar_real in train_loader:
            '''
            enc_inputs: [batch_size, src_len]
            dec_inputs: [batch_size, tgt_len]
            dec_outputs: [batch_size, tgt_len]
            '''
            # input, tar_input, tar_real = enc_inputs.cuda(), dec_inputs.cuda(), dec_outputs.cuda()
            # tar_input = target[:, :-1]
            # # 倒数第2个
            # tar_real = target[:, 1:]
            # # 第1个到最后一个
            # input, tar_input, tar_real = load_data(input, tar_input, tar_real)
            enc_padding_mask, dec_mask, enc_dec_padding_mask = mu.create_mask(input=input, target=tar_input)
            predictions, dec_attn_weights = trm_model(batch_size=batch_size, inputs=input, target=tar_input,
                                                      enc_padding_mask=enc_padding_mask, dec_mask=dec_mask,
                                                      enc_dec_padding_mask=enc_dec_padding_mask)
            # print(predictions.view(-1, predictions.size(-1)).shape)
            # train_loss = loss_function(predictions.view(-1, predictions.size(-1)), tar_real.view(-1), loss_obj)
            train_loss = loss_obj(predictions.view(-1, predictions.size(-1)), tar_real.view(-1))
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            print('Completed training batch', epoch, 'Training Loss is: %.4f' % train_loss)
            total_train_loss += train_loss.item()
        train_loss_value = total_train_loss / len(train_loader)
        # # Validation Loop
        # with torch.no_grad():
        #     trm_model.eval()
        #     for data in valset:
        #         inputs, outputs = data
        #         predicted_outputs = trm_model(inputs)
        #         val_loss = loss_fn(predicted_outputs, outputs)
        #
        #         # The label with the highest value will be our prediction
        #         _, predicted = torch.max(predicted_outputs, 1)
        #         running_vall_loss += val_loss.item()
        #         total += outputs.size(0)
        #         running_accuracy += (predicted == outputs).sum().item()
        #
        #         # Calculate validation loss value
        # val_loss_value = running_vall_loss / len(validate_loader)
        #
        # # Calculate accuracy as the number of correct predictions in the validation batch divided by the total number of predictions done.
        # accuracy = (100 * running_accuracy / total)
        #
        # # Save the model if the accuracy is the best
        # if accuracy > best_accuracy:
        #     saveModel()
        #     best_accuracy = accuracy

        # Print the statistics of the epoch
        # print('Completed training batch', epoch, 'Training Loss is: %.4f' % train_loss_value)
    model_path = checkpoint_folder + time.strftime("%Y_%m_%d_%H_%M_%S") + '.pth'
    save_model(trm_model, model_path)


def test(test_loader, model, start_symbol):
    enc_inputs, _, _ = next(iter(test_loader))
    mu = MaskUtils()
    for i in range(len(enc_inputs)):
        input = enc_inputs[i].view(1, -1).to(device)
        enc_padding_mask = mu.create_padding_mask(input)

        greedy_dec_input = greedy_decoder(model, input, start_symbol=start_symbol,
                                          enc_padding_mask=enc_padding_mask)
        predict, _, _, _ = model(input, greedy_dec_input)

        predict = predict.data.max(1, keepdim=True)[1]
        print(enc_inputs[i], '->', [idx2word[n.item()] for n in predict.squeeze()])


def evaluate(input, trm_model):
#     input_id = [pt_tokenizer.vocab_size] + pt_tokenizer.encode(input) + [pt_tokenizer.vocab_size + 1]
#     enc_input = input_id.unsqueeze(0)
#     dec_input = en_tokenizer.vocab_size.unsqueeze(0)
#     mu = MaskUtils()
#     for i in range(max_len):
#         enc_padding_mask, dec_mask, enc_dec_padding_mask = mu.create_mask(input=enc_input, target=dec_input)
#         predictions, dec_attn_weights = trm_model.eval(batch_size=batch_size, inputs=enc_input, target=dec_input,
#                                                        enc_padding_mask=enc_padding_mask, dec_mask=dec_mask,
#                                                        enc_dec_padding_mask=enc_dec_padding_mask)
#         predictions = predictions[:-1, :]
#         # 获取最后一个预测值
#         predictions_id = torch.argmax(predictions, dim=-1).type(torch.int32)
#         if predictions_id.eq(en_tokenizer.vocab_size + 1):
#             return dec_input.unsqueeze(0), dec_attn_weights
#         dec_input = torch.cat(dec_input, predictions_id)
#     return dec_input.unsqueeze(0), dec_attn_weights
    pass

if __name__ == '__main__':
    sentences = [
        # enc_input           dec_input         dec_output
        ['ich mochte ein bier P', 'S i want a beer .', 'i want a beer . E'],
        ['ich mochte ein cola P', 'S i want a coke .', 'i want a coke . E']
    ]

    # Padding Should be Zero
    # 构建词表
    src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4, 'cola': 5}
    src_vocab_size = len(src_vocab)

    tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'coke': 5, 'S': 6, 'E': 7, '.': 8}
    idx2word = {i: w for i, w in enumerate(tgt_vocab)}
    tgt_vocab_size = len(tgt_vocab)
    enc_inputs, dec_inputs, dec_outputs = make_data(sentences, src_vocab, tgt_vocab)
    # # print(enc_inputs)
    # # print(dec_inputs)
    # # print(dec_outputs)
    # train_loader = data.DataLoader(DataSet(enc_inputs, dec_inputs, dec_outputs), batch_size, True)
    # trainset =,
    # valset =,
    # val_loader = data.DataLoader(DataSet(enc_inputs, dec_inputs, dec_outputs), 2, True)
    # train(trainset=None)
    train_loader = data.DataLoader(DataSet(enc_inputs, dec_inputs, dec_outputs), batch_size, True)
    trm_model = Transformer(layer_nums=layer_nums, input_vocab_size=input_vocab_size,
                            target_vocab_size=target_vocab_size,
                            max_len=max_len, dim_model=dim_model, head_nums=head_nums, dff=dff)
    load_model(trm_model)
    train(train_loader, trm_model)
    # test(train_loader, trm_model, start_symbol=tgt_vocab["S"])
    # tokenizer = get_tokenizer('basic_english')
    # evaluate(train_loader, trm_model)
