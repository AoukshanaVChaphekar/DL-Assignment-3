import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from dataPreprocessing import DataProcessing
from model import Encoder
from model import Decoder
from train import trainIters
from train import train

if __name__ == "__main__":

    # Set to device to cuda,if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparamters
    config = {
        "hidden_size" : 256,
        "source_lang" : 'en',
        "target_lang" : 'hin',
        "cell_type"   : "gru",
        "numEncoders" : 1,
        "numDecoders" : 1,
        "drop_out"    : 0.1, 
        "embedding_size" : 256,
        "bidirectional" : False, # TODO in Attention
    }

    # Load and pre-process data
    data = DataProcessing(DATAPATH = 'aksharantar_sampled', source_lang = config["source_lang"], target_lang = config["target_lang"],device = device)
    
    # Create encoder with input size = number of characters in source langauge and specified embedding size
    encoder = Encoder(data.num_decoder_tokens,config["embedding_size"],config).to(device)
    
    # Create encoder with output size = number of characters in target langauge and specified embedding size
    decoder = Decoder(config["embedding_size"],data.num_decoder_tokens,config).to(device)
    
    # Train the model
    trainIters(data = data,encoder = encoder,decoder = decoder,n_iters = data.number_of_train_samples // 25,device=device,max_length = data.getMaxLength())
   

