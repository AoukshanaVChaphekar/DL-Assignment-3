import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from dataPreprocessing import DataProcessing
from model import Encoder
from model import Decoder
from trainModel import trainIters
from torch.utils.data import DataLoader
import time

import argparse


'''
Run this file if you want to run the code without WANDB
'''

def trainForConfigs(config,add_wandb):
    # Load and pre-process data
    device = config["device"]
    data = DataProcessing(DATAPATH = 'aksharantar_sampled', source_lang = config["source_lang"], target_lang = config["target_lang"],device = device,config = config)
    
    config["maxLength"] = data.getMaxLength()
    batch_size = config["batch_size"]
    
    # Create encoder with input size = number of characters in source langauge and specified embedding size
    encoder = Encoder(data.num_encoder_tokens,config).to(device)
    
    # Create encoder with output size = number of characters in target langauge and specified embedding size
    decoder = Decoder(data.num_decoder_tokens,config,data).to(device)
    

    trainLoader,total_batches = getTrainLoader(data,batch_size)    
    
    # Train the model and compute loss and accuracy
    trainIters(config = config,loader=trainLoader,total_batches=total_batches,data = data,encoder = encoder,decoder = decoder,wandbapply = add_wandb)
    
# Returns loader for train data and total number of batches in training data
def getTrainLoader(data,batch_size):
        trainPairs = data.createTrainPairs()
        training_pairs = []
        for pair in trainPairs:
            training_pairs.append(data.tensorsFromPair(pair))
        
        trainLoader = DataLoader(training_pairs,batch_size = batch_size,shuffle = True)
        total_batches = len(training_pairs) // batch_size
   
        return trainLoader,total_batches

def update_parameters(config):

        parser = argparse.ArgumentParser(description='DL Assignment 3 Parser')

        parser.add_argument('-wp', '--wandb_project',
                            type=str, metavar='', help='wandb project')
        
        parser.add_argument('-we', '--wandb_entity', type=str,
                            metavar='', help='wandb entity')
        
        parser.add_argument('-hs', '--hidden_size', type=int,
                            metavar='', help='hidden_size')
        
        parser.add_argument('-c', '--cell_type', type=str,
                            metavar='', help='cell_type')
        
        parser.add_argument('-nl', '--numLayers', type=int,
                            metavar='', help='numLayers')
        
        parser.add_argument('-dp', '--drop_out', type=float,
                            metavar='', help='drop_out')
        
        parser.add_argument('-es', '--embedding_size',
                            type=int, metavar='', help='embedding_size')
        
        parser.add_argument('-bs', '--batch_size',
                            type=int, metavar='', help='batch_size')
        
        parser.add_argument('-e', '--epoch',
                            type=int, metavar='', help='epoch')
        
        parser.add_argument('-lr', '--learning_rate',
                            type=float, metavar='', help='learning rate')
        
        args = parser.parse_args()

        if (args.wandb_project != None):
            config["wandb_project"] = args.wandb_project
        
        if (args.wandb_entity != None):
            config["wandb_entity"] = args.wandb_entity
        
        if (args.hidden_size != None):
            config["hidden_size"] = args.hidden_size
        
        if (args.cell_type != None):
            config["cell_type"] = args.cell_type
        
        if (args.numLayers != None):
            config["numLayers"] = args.numLayers
        
        if (args.drop_out != None):
            config["drop_out"] = args.drop_out
        
        if (args.embedding_size != None):
            config["embedding_size"] = args.embedding_size
        
        if (args.batch_size != None):
            config["batch_size"] = args.batch_size
        
        if (args.epoch != None):
            config["epoch"] = args.epoch
        
        if (args.learning_rate != None):
            config["learning_rate"] = args.learning_rate
  

if __name__ == "__main__":

    # Set to device to cuda if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = {
        "wandb_project": "DL Assignment 3-1",
        "wandb_entity": "cs22m019",
        "hidden_size" : 512,
        "source_lang" : 'en',
        "target_lang" : 'hin',
        "cell_type"   : "lstm",
        "numLayers" : 3,
        "drop_out"    : 0.3, 
        "embedding_size" : 256,
        "bidirectional" : False,
        "batch_size" : 64,
        "attention" : False,
        "epoch" : 10,
        "device" : device,
        "learning_rate" : 0.001
    }

    # Update parameters obtained from command line
    update_parameters(config)
    
    startime = time.time()
    trainForConfigs(config,add_wandb=False)
    endTime = (time.time() - startime)
    print(endTime / 60)

        

       