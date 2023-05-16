import torch
import torch.nn as nn
from torch import optim
import random
from torch.autograd import Variable
import time
import wandb
import numpy as np
import csv
import matplotlib.pyplot as plt
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
def trainforOneEpoch(config,data,source_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    
    device = config["device"]
    max_length = data.max_length
    batch_size = config["batch_size"]
    attention = config["attention"]
    hidden_size = config["hidden_size"]
    
    
    # Initailize initial hidden layer for encoder
    encoder_hidden = encoder.initHidden(device,config["numLayers"])
    
    if config["cell_type"] == "lstm":
        encoder_cell_state = encoder.initHidden(device,config["numLayers"])
        encoder_hidden = (encoder_hidden,encoder_cell_state)
    
    # Empty the gradients for encoder and decoder
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    source_tensor = source_tensor.squeeze()
    target_tensor = target_tensor.squeeze()
    
    # Length of source and target tensor
    source_length = source_tensor.size(0)
    target_length = target_tensor.size(0)
    
    
    # Stores all encoder outputs for each character in source word
    if attention:
        encoder_outputs = torch.zeros(max_length + 1,batch_size, hidden_size, device = device)

    # Initialize loss to ZERO
    loss = 0
    
    # encoder encodes each character index in source_word
    for ei in range(source_length):
        encoder_output, encoder_hidden = encoder(source_tensor[ei], encoder_hidden)
        if attention:
            encoder_outputs[ei] = encoder_output[0]
    

    # Initialize decoder input with start of word token
    decoder_input = torch.tensor([[data.SOW_char2int] * batch_size],device = device)

    # initial hidden layer for decoder will be final hidden layer from the encoder
    decoder_hidden = encoder_hidden
    
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
   
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            if attention:  
                decoder_output, decoder_hidden,decoder_attention = decoder(decoder_input, decoder_hidden,encoder_outputs.reshape(batch_size,max_length + 1,hidden_size))
            else:
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)

            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            if attention:
                decoder_output, decoder_hidden,decoder_attention = decoder(decoder_input, decoder_hidden,encoder_outputs.reshape(batch_size,max_length + 1,hidden_size))
            else:
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            
            # returns top value and index from the decoder output
            topv, topi = decoder_output.topk(1)
            
            # Squeeze all the dimensions that are 1 and returns a new tensor detatched from the current history graph
            decoder_input = topi.squeeze().detach()

            # Compute loss
            loss += criterion(decoder_output, target_tensor[di])

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
def trainIters(config,total_batches,loader,data,encoder,decoder,wandbapply):

    if wandbapply:
        wandb.init(
            project=config["wandb_project"]
        )

    epochs = config["epoch"]
    learning_rate = config["learning_rate"]
    criterion = nn.NLLLoss()
    
    # Set optimizers for encoder and decoder
    encoder_optimizer = optim.NAdam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.NAdam(decoder.parameters(), lr=learning_rate)
    
    
    for epoch in range(epochs):
        epoch_loss = 0
        batch_no = 1

        # Train for each batch
        for batchx,batchy in loader:
               
            batchx = batchx.transpose(0,1)
            batchy = batchy.transpose(0,1)
            batch_loss = trainforOneEpoch(config = config,
                                            data = data,
                                            source_tensor = batchx, 
                                            target_tensor = batchy, 
                                            encoder = encoder,
                                            decoder = decoder,
                                            encoder_optimizer = encoder_optimizer, 
                                            decoder_optimizer = decoder_optimizer, 
                                            criterion = criterion)
            
            epoch_loss += batch_loss
            batch_no+=1
        
        
        val_loader = data.getValLoader()
        
        if epoch == (epochs - 1):
            
            # Compute validation accuracy
            validation_loss ,validation_accuracy = evaluate(config=config,
                loader=val_loader,
                data=data,
                encoder=encoder,
                decoder=decoder,
                training_completed=True
            )
        else:
            # Compute validation accuracy
            validation_loss ,validation_accuracy = evaluate(config=config,
                loader=val_loader,
                data=data,
                encoder=encoder,
                decoder=decoder,
                training_completed=False
            )
        train_loss = epoch_loss / total_batches
       
        print("epoch:{epoch}, train loss:{train_l}, validation loss:{validation_l}, validation accuracy:{validation_ac}".\
                  format(epoch = epoch + 1,train_l = train_loss,validation_l = validation_loss,validation_ac = validation_accuracy))

        if wandbapply:
            wandb.log({'train loss':train_loss,'validation loss':validation_loss, 'validation accuracy':validation_accuracy})

    # test_loader = data.getTestLoader()
        
    # config["batch_size"] = 1
    # test_loss ,test_accuracy = evaluate(config=config,
    #          loader=test_loader,
    #          data=data,
    #          encoder=encoder,
    #          decoder=decoder,
    #     )
    # print("Test accuracy:",test_accuracy)   

# Compute accuracy 
def evaluate(config,data, loader, encoder, decoder,training_completed) :

        loss = 0
        totalCorrectWords = 0
        batchNumber = 1
        batch_size = config["batch_size"]
        
        totalWords = len(loader.sampler)
        totalBatches = len(loader.sampler) // batch_size

        # Loss Function
        criterion = nn.NLLLoss()

        for sourceTensor, targetTensor in loader :
            batchLoss, correctWords,attentions = evaluateOneBatch(config,data,sourceTensor, targetTensor, encoder, decoder, criterion)

            loss += batchLoss
            totalCorrectWords += correctWords

        # If training is completed,then dispay heatmaps
        if training_completed == True :
            if attentions is not None:
                volume  = attentions.numpy()

                for point in range(10):
                    heatMap = np.zeros((data.max_length + 1,data.max_length + 1))

                    for i in range(data.max_length + 1):
                        for k in range(data.max_length + 1):
                            heatMap[i][k] = volume[i][point][k]


                    plotHeatMap(heatMap)

        return (loss / totalBatches), (totalCorrectWords / totalWords) * 100

def plotHeatMap(heatMap):
    plt.imshow(heatMap,cmap = 'magma')
    plt.title("HeatMap")
    plt.colorbar()
    plt.show()

def evaluateOneBatch(config,data, sourceTensorBatch, targetTensorBatch, encoder, decoder, criterion) :

        loss = 0
        correctWords = 0

        batchSize = data.batch_size
        device = config["device"]
        maxLengthWord = data.max_length
        cell_type = config["cell_type"]
        attention = config["attention"]
        hidden_size = config["hidden_size"]

        sourceTensor = Variable(sourceTensorBatch.transpose(0, 1))
        targetTensor = Variable(targetTensorBatch.transpose(0, 1))
        
        # Get source length
        sourceTensorLength = sourceTensor.size()[0]
        targetTensorLength = targetTensor.size()[0]

        predictedBatchOutput = torch.zeros(targetTensorLength, batchSize, device = device)

        # Initialize initial hidden state of encoder
        encoderHidden = encoder.initHidden(device = device,numLayers = config["numLayers"])

        if cell_type == "lstm":
            encoderCell = encoder.initHidden(device = device,numLayers = config["numLayers"])
            encoderHidden = (encoderHidden, encoderCell)

        if attention:
            encoderOutputs = torch.zeros(maxLengthWord + 1, batchSize, hidden_size, device = device)

        for ei in range(sourceTensorLength):
            encoderOutput, encoderHidden = encoder(sourceTensor[ei], encoderHidden)

            if attention :
                encoderOutputs[ei] = encoderOutput[0]

        # Initialize input to decoder with start of word token
        decoderInput = torch.tensor([[data.SOW_char2int] * batchSize], device = device)

        # initial hidden state for decoder will be final hidden state of encoder
        decoderHidden = encoderHidden

        if attention :
            decoderAttentions = torch.zeros(maxLengthWord + 1,batchSize, maxLengthWord + 1)
        
        for di in range(targetTensorLength):
            if attention :
                # Pass the decoderInput, decoderHidden and encoderOutputs to the decoder
                decoderOutput, decoderHidden, decoderAttention = decoder(decoderInput, decoderHidden, encoderOutputs.reshape(batchSize,maxLengthWord + 1,hidden_size))
                decoderAttentions[di] = decoderAttention.data
            else : 
                decoderOutput, decoderHidden = decoder(decoderInput, decoderHidden)
            
            loss += criterion(decoderOutput, targetTensor[di].squeeze())
            
            topv, topi = decoderOutput.data.topk(1)
            decoderInput = torch.cat(tuple(topi))
            predictedBatchOutput[di] = torch.cat(tuple(topi))
    
        predictedBatchOutput = predictedBatchOutput.transpose(0,1)

        ignore = [data.SOW_char2int, data.EOW_char2int,data.PAD_char2int]

        predicted_list = []
        actual_list = []

        for di in range(predictedBatchOutput.size()[0]):

            predicted = [letter.item() for letter in predictedBatchOutput[di] if letter not in ignore]
            actual = [letter.item() for letter in targetTensorBatch[di] if letter not in ignore]
            
            predicted_list.append(predicted)
            actual_list.append(actual)
            
            if predicted == actual:
                correctWords += 1
        
        # writeToCSV(predicted_list,actual_list)
        
        if attention:
            return loss.item() / len(sourceTensorBatch), correctWords,decoderAttentions
        return loss.item() / len(sourceTensorBatch), correctWords,None

def writeToCSV(predicted_list,actual_list):

    fields = ["Actual","Predicted"]

    rows = []
    for i in range(len(predicted_list)):
        rows.append([actual_list[i],predicted_list[i]])
    
    filename = "predcitions_vanilla.csv"

    # writing to csv file 
    with open(filename, 'w') as csvfile: 
        # creating a csv writer object 
        csvwriter = csv.writer(csvfile) 
            
        # writing the fields 
        csvwriter.writerow(fields) 
            
        # writing the data rows 
        csvwriter.writerows(rows)

