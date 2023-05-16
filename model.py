import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    
    # Encoder Destructor
    def __init__(self, input_size,config):
        super(Encoder, self).__init__()

        # Store parameters in class varaibles
        self.hidden_size = config["hidden_size"]
        self.embedding_size = config["embedding_size"]
        self.cell_type = config["cell_type"] 
        self.numLayers = config["numLayers"]
        self.drop_out = config["drop_out"]
        self.bidirectional = config["bidirectional"]
        self.batch_size = config["batch_size"]
        
        # input_size - contains the number of encoder tokens which is input to Embedding
        # hidden_size - size of each embedding vector
        # Create an Embedding for the Input 
        # Each character will have an embedding of size = embedding_size
        self.embedding = nn.Embedding(input_size, self.embedding_size)
        self.dropout = nn.Dropout(self.drop_out)
        
        # the cell_type - GRU
        if self.cell_type == "gru":
            ''' Input to GRU -  1.number of expected features in x - embedded input 
                                2.number of features in hidden state - hidden_size
                                3.number of layers (stacking GRUs together) '''
            self.gru = nn.GRU(self.embedding_size, self.hidden_size,num_layers = self.numLayers,dropout = self.drop_out,bidirectional = self.bidirectional)
            self.rnnLayer = self.gru
        
        # the cell_type - RNN
        if self.cell_type == "rnn":
            ''' Input to RNN -  1.number of expected features in x
                                2.number of features in hidden state
                                3.number of layers (stacking RNNs together) '''
            
            self.rnn = nn.RNN(self.embedding_size,self.hidden_size,num_layers = self.numLayers,dropout = self.drop_out,bidirectional = self.bidirectional)
            self.rnnLayer = self.rnn
        
        # the cell_type - LSTM
        if self.cell_type == "lstm":
            ''' Input to LSTM - 1.number of expected features in x
                                2.number of features in hidden state
                                3.number of layers (stacking LSTMs together) '''
            
            self.lstm = nn.LSTM(self.embedding_size,self.hidden_size,num_layers = self.numLayers,dropout = self.drop_out,bidirectional = self.bidirectional)
            self.rnnLayer = self.lstm

    # Encoder forward pass
    def forward(self, input, hidden,cell_state = None):
        '''Input -> hidden      - initial hidden state for each element in the input sequence
                    cell_hidden - the initial cell state for each element in the input sequence
        '''
       
        # Creates a embedded tensor by passing the input to the embedding layer and resizing the output to (1,batch_size,-1)
        embedded = self.dropout(self.embedding(input).view(1, self.batch_size, -1))
        
        # Pass this embedded input to the GRU/LSTM/RNN model
        output = embedded
          
        '''Output     -     1.Output features from the last layer
                            2.final hidden state for each element which is passed to decoder as a context vector'''
        output, hidden = self.rnnLayer(output, hidden)
        return output, hidden
        
        
    # Initailizes initial hidden layer for encoder
    def initHidden(self,device,numLayers):
        if self.bidirectional:
            return torch.zeros(numLayers * 2, self.batch_size, self.hidden_size, device=device)
        else:
            return torch.zeros(numLayers, self.batch_size, self.hidden_size, device=device)

class Decoder(nn.Module):

    # Decoder Constructor
    def __init__(self,output_size,config,data): 
        super(Decoder, self).__init__()

        
        # Store parameters in class varaibles
        self.numLayers = config["numLayers"]
        self.cell_type = config["cell_type"]
        self.hidden_size = config["hidden_size"]
        self.embedding_size = config["embedding_size"]
        # Create embedding for input
        self.embedding = nn.Embedding(output_size, self.embedding_size)
        self.drop_out = config["drop_out"]       
        self.bidirectional = config["bidirectional"]
        self.batch_size = config["batch_size"]
        self.dropout = nn.Dropout(self.drop_out)
        
        if self.cell_type == "gru":
            self.gru = nn.GRU(self.embedding_size, self.hidden_size,num_layers = self.numLayers,dropout = self.drop_out,bidirectional = self.bidirectional)
            self.rnnLayer = self.gru

        if self.cell_type == "rnn":
            self.rnn = nn.RNN(self.embedding_size,self.hidden_size,self.numLayers,dropout = self.drop_out,bidirectional = self.bidirectional)
            self.rnnLayer = self.rnn
        
        if self.cell_type == "lstm":
            self.lstm = nn.LSTM(self.embedding_size,self.hidden_size,self.numLayers,dropout = self.drop_out,bidirectional = self.bidirectional)
            self.rnnLayer = self.lstm
       
        # Creating a dense layer
        if self.bidirectional:
            self.out = nn.Linear(self.hidden_size * 2 ,output_size)
        else:
            self.out = nn.Linear(self.hidden_size, output_size)
        
        # Softmax function as an output function
        self.softmax = nn.LogSoftmax(dim = 1)

    def forward(self, input, hidden,cell_state = None):

        # Create an embedding for input and resize it to (1,batch_size,-1)
        # The output of embedding layer is passed as an input to decoder
        output = self.dropout(self.embedding(input).view(1, self.batch_size, -1))

        # Applying ReLU activation function
        output = F.relu(output)

        # Pass output and previous hidden state to model RNN/LSTM/GRU
        output, hidden = self.rnnLayer(output, hidden)
        
        # apply softmax function as an output function
        output = self.softmax(self.out(output[0]))
        
        return output, hidden
               
class DecoderAttention(nn.Module) :

    # Decoder Constructor
    def __init__(self, output_size,configs,data) :

        super(DecoderAttention, self).__init__()

        
        # Store parameters in class varaibles
        self.hidden_size = configs['hidden_size']
        self.embedding_size = configs['embedding_size']
        self.cell_type = configs['cell_type']
        self.device = configs['device'] 
        self.numLayers = configs['numLayers']
        self.dropout_rate = configs['drop_out']
        self.maxLengthWord = data.getMaxLength()
        self.maxLengthTensor = self.maxLengthWord + 1
        self.batch_size = configs['batch_size']
        self.bidirectional = configs['bidirectional']

        # Create an Embedding for the Input
        self.embedding = nn.Embedding(num_embeddings = output_size, embedding_dim = self.embedding_size)

        # Attention Layer
        self.attention_layer = nn.Linear(self.embedding_size + self.hidden_size, self.maxLengthTensor)

        # Combine Embedded and Attention Applied Outputs
        self.attention_combine = nn.Linear(self.embedding_size + self.hidden_size, self.embedding_size)

        # Dropout Layer
        self.dropout = nn.Dropout(self.dropout_rate)

        if self.cell_type == "gru" :
            self.RNNLayer = nn.GRU(self.embedding_size, self.hidden_size, num_layers = self.numLayers, dropout = self.dropout_rate, bidirectional = self.bidirectional)

        elif self.cell_type == "rnn" :
            self.RNNLayer = nn.RNN(self.embedding_size, self.hidden_size, num_layers = self.numLayers, dropout = self.dropout_rate, bidirectional = self.bidirectional)
        
        else : 
            self.RNNLayer = nn.LSTM(self.embedding_size, self.hidden_size, num_layers = self.numLayers, dropout = self.dropout_rate, bidirectional = self.bidirectional)

        # Create a linear layer
        if self.bidirectional:
            self.out = nn.Linear(2 * self.hidden_size, output_size)
        else :
            self.out = nn.Linear(self.hidden_size, output_size)

    # Decoder Forward Pass
    def forward(self, input, hidden, encoder_outputs) :

        # Pass the Input through the Embedding layer to get embedded input 
        # The embedded input is reshaped to have a shape of (1, batch_size, -1)
        embedded = self.embedding(input).view(1, self.batch_size, -1)

        embedded = self.dropout(embedded)

        if self.cell_type == "lstm" :
            # Calculate Attention Weights
            embeddedHidden = torch.cat((embedded[0], hidden[0][0]), 1)
            embeddedHiddenAttention = self.attention_layer(embeddedHidden)
            attentionWeights = F.softmax(embeddedHiddenAttention, dim = 1)   
        else :
            # Calculate Attention Weights
            attentionWeights = F.softmax(self.attention_layer(torch.cat((embedded[0], hidden[0]), 1)), dim = 1)

        
        attentionApplied = torch.bmm(attentionWeights.view(self.batch_size, 1, self.maxLengthTensor), encoder_outputs).view(1, self.batch_size, -1)

        output = torch.cat((embedded[0], attentionApplied[0]), 1)

        # Pass the output through the attention combine layer
        output = self.attention_combine(output).unsqueeze(0)

        # Apply ReLU activation
        output = F.relu(output)
        output, hidden = self.RNNLayer(output, hidden)
        
        # Apply softmax to the output of out linear layer
        output = F.log_softmax(self.out(output[0]), dim = 1)

        # Return the output, hidden state and attention weights
        return output, hidden, attentionWeights