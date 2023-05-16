import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from dataPreprocessing import DataProcessing
from model import Encoder
from model import Decoder
from model import AttnDecoder,DecoderAttention
from trainModel import trainIters
from torch.utils.data import DataLoader
import time


'''
Run this file if you want to run the code without WANDB
'''

def trainForConfigs(config):
    # Load and pre-process data
    device = config["device"]
    data = DataProcessing(DATAPATH = 'aksharantar_sampled', source_lang = config["source_lang"], target_lang = config["target_lang"],device = device,config = config)
    
    config["maxLength"] = data.getMaxLength()
    batch_size = config["batch_size"]
    
    # Create encoder with input size = number of characters in source langauge and specified embedding size
    encoder = Encoder(data.num_decoder_tokens,config).to(device)
    
    # Create encoder with output size = number of characters in target langauge and specified embedding size
    if config["attention"]:
        decoder = DecoderAttention(data.num_decoder_tokens,config,data).to(device)
    else:
        decoder = Decoder(data.num_decoder_tokens,config,data).to(device)
    

    trainLoader,total_batches = getTrainLoader(data,batch_size)    
    
    # Train the model and compute loss and accuracy
    trainIters(config = config,loader=trainLoader,total_batches=total_batches,data = data,encoder = encoder,decoder = decoder)
    
# Returns loader for train data and total number of batches in training data
def getTrainLoader(data,batch_size):
        trainPairs = data.createTrainPairs()
        training_pairs = []
        for pair in trainPairs:
            training_pairs.append(data.tensorsFromPair(pair))
        
        trainLoader = DataLoader(training_pairs,batch_size = batch_size,shuffle = True)
        total_batches = len(training_pairs) // batch_size
   
        return trainLoader,total_batches
  

if __name__ == "__main__":

    # Set to device to cuda if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    config = {
        "hidden_size" : 512,
        "source_lang" : 'en',
        "target_lang" : 'hin',
        "cell_type"   : "lstm",
        "numLayers" : 3,
        "drop_out"    : 0, 
        "embedding_size" : 256,
        "bidirectional" : False,
        "batch_size" : 128,
        "attention" : True,
        "epoch" : 10,
        "device" : device,
        "learning_rate" : 0.001
    }

    startime = time.time()
    trainForConfigs(config)
    endTime = (time.time() - startime)
    print(endTime / 60)

