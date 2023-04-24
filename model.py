import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    '''
    Input : 1.input_size - Number of characters in source language
            2.hidden_size - Size of embedding for each character,
                            Size of Input for GRU / RNN / LSTM,
                            Size of Hidden State for GRU / RNN / LSTM,
    '''

    # Encoder Destructor
    def __init__(self, input_size, hidden_size,config):
        super(Encoder, self).__init__()

        # Number of features in hidden state ,same as number of expected features in input x
        self.hidden_size = hidden_size

        # Cell_type -> RNN,LSTM,GRU
        self.cell_type = config["cell_type"] 

        # Stores the number of encoders
        self.numEncoders = config["numEncoders"]

        self.drop_out = config["drop_out"]
        
        # input_size - contains the number of encoder tokens which is input to Embedding
        # hidden_size - size of each embedding vector
        # Create an Embedding for the Input 
        # Each character will have an embedding of size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        
        # the cell_type - GRU
        if self.cell_type == "gru":
            ''' Input to GRU -  1.number of expected features in x - embedded input of size hidden_size
                                2.number of features in hidden state - hidden_size
                                3.number of layers (stacking GRUs together) '''
            self.gru = nn.GRU(hidden_size, hidden_size,num_layers = self.numEncoders,dropout = self.drop_out)
        
        # the cell_type - RNN
        if self.cell_type == "rnn":
            ''' Input to RNN -  1.number of expected features in x
                                2.number of features in hidden state
                                3.number of layers (stacking GRUs together) '''
            
            self.rnn = nn.RNN(hidden_size,hidden_size,num_layers = self.numEncoders,drop_out = self.dropout)
        
        # the cell_type - LSTM
        # if self.cell_type == "lstm":
        #     ''' Input to LSTM - 1.number of expected features in x
        #                         2.number of features in hidden state
        #                         3.number of layers (stacking GRUs together) '''
            
        #     self.lstm = nn.LSTM(hidden_size,hidden_size,num_layers = self.numEncoders)

    # Encoder forward pass
    def forward(self, input, hidden,cell_state = None):
        '''Input -> hidden      - initial hidden state for each element in the input sequence
                    cell_hidden - the initial cell state for each element in the input sequence
        '''
       
        # Creates a embedded tensor by passing the input to the embedding layer and resizing the output to (1,1,-1)
        embedded = self.embedding(input).view(1, 1, -1)
        
        # Pass this embedded input to the GRU/LSTM/RNN model
        output = embedded
          
        if self.cell_type == "gru":
            '''Output of GRU - 1.Output features from the last layer
                               2.final hidden state for each element which is passed to decoder as a context vector'''
            output, hidden = self.gru(output, hidden)
            return output, hidden
        
        if self.cell_type == "rnn":
            '''Output of RNN - 1.Output features from the last layer
                               2.final hidden state for each element which is passed to decoder as a context vector'''
            output, hidden = self.rnn(output, hidden)
            return output, hidden
        
        # if self.cell_type == "lstm":
        #     '''Output of LSTM - 1.Output features from the last layer
        #                         2.final hidden state for each element which is passed to decoder as a context vector
        #                         3. final cell state for each element '''
        #     output, hidden,cell_state = self.lstm(output, hidden,cell_state)
        #     return output, hidden,cell_state

    # Initailizes initial hidden layer for encoder
    def initHidden(self,device):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class Decoder(nn.Module):
    
    '''
    Input:  1.hidden_size -  Size of embedding for each character,
                            Size of Input for GRU / RNN / LSTM,
                            Size of Hidden State for GRU / RNN / LSTM.
            2.output_size - number of characters in target language
    '''
    # Decoder Constructor
    def __init__(self, hidden_size, output_size,config): 
        super(Decoder, self).__init__()

        # Stores the number of decoders
        self.numDecoders = config["numDecoders"]

        # cell_type -> RNN,LSTM,GRU
        self.cell_type = config["cell_type"]
        
        self.hidden_size = hidden_size
        # Create embedding for input
        self.embedding = nn.Embedding(output_size, hidden_size)

        # Number of neurons in hidden layer 
        self.hidden_neurons = config["hidden_size"]
        
        self.drop_out = config["drop_out"]
        
        if self.cell_type == "gru":
            self.gru = nn.GRU(hidden_size, hidden_size,num_layers = self.numDecoders,dropout = self.drop_out)
        
        if self.cell_type == "rnn":
            self.rnn = nn.RNN(hidden_size,hidden_size,self.numDecoders,dropout = self.drop_out)

        # if self.cell_type == "lstm":
        #     self.lstm = nn.LSTM(hidden_size,hidden_size,self.numDecoders)

        # TODO check number of neurons
        # Creating a dense layer
        self.out = nn.Linear(hidden_size, output_size)

        # Softmax function as an output function
        self.softmax = nn.LogSoftmax(dim = 1)

    def forward(self, input, hidden,cell_state = None):

        # Create an embedding for input and resize it to (1,1,-1)
        # The output of embedding layer is passed as an input to decoder
        output = self.embedding(input).view(1, 1, -1)

        # Applying ReLU activation function
        output = F.relu(output)

        if self.cell_type == "gru":
            # Pass output and previous hidden state to GRU
            output, hidden = self.gru(output, hidden)
            
            # apply softmax function as an output function
            output = self.softmax(self.out(output[0]))
            
            return output, hidden

        if self.cell_type == "rnn":
            # Pass output and previous hidden state to RNN
            output,hidden = self.rnn(output,hidden)
            
            # apply softmax function as an output function
            output = self.softmax(self.out(output[0]))
            
            return output, hidden

        # if self.cell_type == "lstm":
        #     output,hidden,cell_state = self.lstm(output,hidden,cell_state)
        #     output = self.softmax(self.out(output[0]))
        #     return output,hidden,cell_state

    # Intialize initial hidden layer
    def initHidden(self,device):
        return torch.zeros(1, 1, self.hidden_size, device=device)