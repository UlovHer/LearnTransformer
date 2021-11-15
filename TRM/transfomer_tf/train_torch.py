import time

import torch
from torch import Tensor, optim
from torch.utils import data

from config.config import layer_nums, max_len, dim_model, head_nums, dff, checkpoint_folder, epochs, batch_size, \
    input_vocab_size, target_vocab_size
from model.transformer import Transformer
from utils.data_utils import make_data, DataSet
from utils.mask_utils import MaskUtils


def loss_function(real: Tensor, predict: Tensor):
    loss_obj = torch.nn.CrossEntropyLoss(reduction="None")
    mask = torch.logical_not(real.eq(0))
    # real和0比较，mask中padding处为0，
    loss = loss_obj(real, predict)
    mask = mask.type(loss.dtype)
    loss *= mask
    return torch.mean(loss)


def load_model(model: Transformer):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device\n")
    model.to(device)  # Convert model parameters and buffers to CPU or Cuda


def save_model(model: Transformer, path: str):
    torch.save(model.state_dict(), path)


def train(trainset=None, valset=None):
    trm_model = Transformer(layer_nums=layer_nums, input_vocab_size=input_vocab_size,
                            target_vocab_size=target_vocab_size,
                            max_len=max_len, dim_model=dim_model, head_nums=head_nums, dff=dff)
    load_model(trm_model)
    optimizer = optim.SGD(trm_model.parameters(), lr=1e-3, momentum=0.99)

    # val_loader = data_loader()
    for epoch in range(epochs):
        total_train_loss = 0.0
        total_accuracy = 0.0
        total_vall_loss = 0.0
        for input, tar_input, tar_real in trainset:
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
            mu = MaskUtils()
            enc_padding_mask, dec_mask, enc_dec_padding_mask = mu.create_mask(input=input, target=tar_input)
            predictions, dec_attn_weights = trm_model(batch_size=batch_size, inputs=input, target=tar_input,
                                                      enc_padding_mask=enc_padding_mask, dec_mask=dec_mask,
                                                      enc_dec_padding_mask=enc_dec_padding_mask)
            train_loss = loss_function(tar_real, predictions)
            train_loss.backward()
            optimizer.step()
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
        print('Completed training batch', epoch, 'Training Loss is: %.4f' % train_loss_value)

    model_path = checkpoint_folder + time.strftime("%Y_%m_d_%H_%M_%S") + '.pth'
    save_model(trm_model, model_path)


if __name__ == '__main__':
    sentences = [
        # enc_input           dec_input         dec_output
        ['ich mochte ein bier P', 'S i want a beer .', 'i want a beer . E'],
        ['ich mochte ein cola P', 'S i want a coke .', 'i want a coke . E']
    ]

    # Padding Should be Zero
    src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4, 'cola': 5}
    src_vocab_size = len(src_vocab)

    tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'coke': 5, 'S': 6, 'E': 7, '.': 8}
    idx2word = {i: w for i, w in enumerate(tgt_vocab)}
    tgt_vocab_size = len(tgt_vocab)
    enc_inputs, dec_inputs, dec_outputs = make_data(sentences, src_vocab, tgt_vocab)
    # print(enc_inputs)
    # print(dec_inputs)
    # print(dec_outputs)
    train_loader = data.DataLoader(DataSet(enc_inputs, dec_inputs, dec_outputs), 2, True)
    # trainset =,
    # valset =,
    # val_loader = data.DataLoader(DataSet(enc_inputs, dec_inputs, dec_outputs), 2, True)
    train(trainset=train_loader)
