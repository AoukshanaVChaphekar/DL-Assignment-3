import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import random
teacher_forcing_ratio = 0.5
'''
    Input ->    1.source_tensor - a tensor for given source word containing character indexes
                2.target_tensor - a tensor for given target word containing character indexes
                3.encoder - object of Encdoer class
                4.decoder - object of Decoder class
                5.encoder_optimizer - optimizer used for encoder
                6.decoder_optimizer - optimizer used for decoder
                7.criterion - loss function
                8.max_length - maximum length of source word
                9.device - CUDA or CPU
'''
def train(data,device,source_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length):
    
    # Initailize initial hidden layer for encoder
    encoder_hidden = encoder.initHidden(device)

    # Empty the gradients for encoder and decoder
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Length of source and target tensor
    source_length = source_tensor.size(0)
    target_length = target_tensor.size(0)

    
    # Stores all encoder outputs for each character in source word
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device = device)

    
    # Initialize loss to ZERO
    loss = 0

    # encoder encodes each character index in source_word
    for ei in range(source_length):
        encoder_output, encoder_hidden = encoder(
            source_tensor[ei], encoder_hidden)
        
        # encoder_outputs[ei] = encoder_output[0, 0]

    # Initialize decoder input with start of word token
    decoder_input = torch.tensor([[data.SOW_char2int]], device = device)

    # initial hidden layer for decoder will be final hidden layer from the encoder
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            
            # returns top value and index from the decoder output
            topv, topi = decoder_output.topk(1)

            # Squeeze all the dimensions that are 1 and returns a new tensor detatched from the current history graph
            decoder_input = topi.squeeze().detach()

            # Compute loss
            loss += criterion(decoder_output, target_tensor[di])

            # break if the EOW is encountered
            if decoder_input.item() == data.EOW_char2int:
                break

    # Backpropagation
    loss.backward()

    # Update parameters
    encoder_optimizer.step()
    decoder_optimizer.step()

    # return the loss
    return loss.item() / target_length



'''
    Input - 1.object of Encoder class
            2.object of Decoder class
            3.n_iters - number of epochs
            4.print_every - prints every given milliseconds
            5.plot_every - plots every given milliseconds
            6.learning rate
'''
def trainIters(data,encoder, decoder, n_iters, device, max_length, print_every = 1000, plot_every = 100, learning_rate = 0.01):
    
    print_every = data.number_of_train_samples // 500
    plot_every = data.number_of_train_samples // 5000
    
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    # Set optimizers for encoder and decoder
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    
    # Create train,validation and test data pairs
    trainPairs = data.createTrainPairs()
    validationPairs = data.createValidationData()
    testPairs = data.createTestData()

    training_pairs = [data.tensorsFromPair(random.choice(trainPairs))
                      for i in range(n_iters)]
    
    # Set loss function
    criterion = nn.NLLLoss()
        
    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        
        # Fetch source_tensor and target_tenosr from the training_pairs
        source_tensor = training_pair[0]
        target_tensor = training_pair[1]

        # Train for train data pair
        loss = train(data = data,
                     device = device,
                     source_tensor = source_tensor, 
                     target_tensor = target_tensor, 
                     encoder = encoder,
                     decoder = decoder,
                     encoder_optimizer = encoder_optimizer, 
                     decoder_optimizer = decoder_optimizer, 
                     criterion = criterion,
                     max_length = max_length)
        
        # accumulate loss
        print_loss_total += loss
        
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print(iter,"/",n_iters,": avg loss - ",print_loss_avg)
            
        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
        
    # Compute validation loss
    computeValidationLoss(validationPairs=validationPairs,
                            data = data,
                            criterion=criterion,
                            encoder=encoder,
                            decoder=decoder,
                            max_length=max_length,
                            device=device)
    # Compute test loss
    computeTestLoss(testPairs=testPairs,
                    data = data,
                    criterion=criterion,
                    encoder=encoder,
                    decoder=decoder,
                    max_length=max_length,
                    device=device)    
        
    showPlot(plot_losses)

import matplotlib.pyplot as plt
# plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    # plt.show()

# Compute validation loss
def computeValidationLoss(validationPairs,data,criterion,encoder,decoder,max_length,device):
    validation_loss = 0
    for i in range(len(validationPairs)):
        source_word = validationPairs[i][0]
        target_word = validationPairs[i][1]
        validation_loss += evaluate(data,criterion,encoder,decoder,source_word,target_word,max_length,device)
        if i % (len(validationPairs) // 10) == 0:
            print("Validation data evaluated:",i,"/",len(validationPairs))

     
    validation_loss = validation_loss / len(validationPairs)
    print("validation loss:",validation_loss)

# Compute test loss
def computeTestLoss(testPairs,data,criterion,encoder,decoder,max_length,device):
    test_loss = 0
    for i in range(len(testPairs)):
        source_word = testPairs[i][0]
        target_word = testPairs[i][1]
        test_loss += evaluate(data,criterion,encoder,decoder,source_word,target_word,max_length,device)
        if i % (len(testPairs) // 10) == 0:
            print("test data evaluated:",i,"/",len(testPairs))

     
    test_loss = test_loss / len(testPairs)
    print("test loss:",test_loss)

# Compute loss 
def evaluate(data,criterion,encoder, decoder, source_word,target_word, max_length,device):

    with torch.no_grad():

        # Generate source tensor from the given input source word
        source_tensor = data.tensorFromWord("source", source_word)
        target_tensor = data.tensorFromWord("target", target_word)

        # Get source length
        source_length = source_tensor.size()[0]
        target_length = target_tensor.size()[0]

        # Initialize initial hidden state of encoder
        encoder_hidden = encoder.initHidden(device)

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(source_length):
            encoder_output, encoder_hidden = encoder(source_tensor[ei],encoder_hidden)
            # encoder_outputs[ei] += encoder_output[0, 0]

        # Initialize input to decoder with start of word token
        decoder_input = torch.tensor([[data.SOW_char2int]], device=device)

        # initial hidden state for decoder will be final hidden state of encoder
        decoder_hidden = encoder_hidden

        # Store characters decoded by decoder in decoded_characters list
        decoded_characters = []
        
        loss = 0
        for di in range(max_length):

            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            # decoder_attentions[di] = decoder_attention.data
            
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == data.EOW_char2int:
                break
            else:
                decoded_characters.append(data.target_int2char[topi.item()])

            if di <  target_length:
                loss += criterion(decoder_output,target_tensor[di])

            decoder_input = topi.squeeze().detach()
        
        decoded_word = ""
        for char in decoded_characters:
            decoded_word += char
        
        return loss.item() / target_length
