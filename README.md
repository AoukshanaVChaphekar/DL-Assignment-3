# DL-Assignment-3

The **wandb report** can be found in the following link:

https://api.wandb.ai/links/cs22m019/x865whmo

## Usage
### To run code with wandb

- **With Attention**
```python
python train_attention.py
```
- **Without Attention**
```python
python train.py
``` 

### To run code without wandb

- **With Attention**
```python
python main_attention.py
```
- **Without Attention**
```python
python main.py
``` 

## Using argparse
| Command Line Argument | Usage |
| --- | --- |
| --wandb_project / -wp  | Name of wandb project |
| --wandb_entity / -we  | Name of wandb entity |
| --hidden_size / -hs  | Hidden size of dense layer |
| --cell_type / -c  | Cell type to use - lstm,gru,rnn |
| --numLayers / -nl  | Number of encoder/decoder layers |
| --drop_out / -dp  | Dropout |
| --embedding_size / -es  | Embedding size |
| --batch_size / -bs  | Batch size |
| --epoch / -e  | Epochs |
| --learning_rate / -lr | Learning Rate |


- **With Attention**
```python
python main_attention.py --wandb_project --wandb_entity --hidden_size --cell_type --numLayers --drop_out --embedding_size --batch_size --epoch --learning_rate
```

OR

```python
python main_attention.py -wp -we -hs -c -nl -dp -es -bs -e -lr
```


- **Without Attention**
```python
python main.py --wandb_project --wandb_entity --hidden_size --cell_type --numLayers --drop_out --embedding_size --batch_size --epoch --learning_rate
```

OR

```python
python main.py -wp -we -hs -c -nl -dp -es -bs -e -lr
```

| Hyperparameter | Values/Usage |
| --- | --- |
| epoch | 5, 10 |
| hidden_size | 128, 256, 512 |
| cell_type | LSTM, RNN, GRU |
| learning_rate | 1e-2, 1e-3 |
| num_layers | 1, 2, 3 |
| drop_out | 0, 0.2, 0.3 |
| embedding_size | 64, 128, 256, 512 |
| batch_size | 32, 64, 128 |
| bidirectional | True, False |


## Files

- dataPreprocessing.py - creates train,test and validation data and preprocesses it
- main.py - execute this file to run the code without attention 
- main_attention.py - execute this file to run the code with attention
- model.py - Contains the Encoder,Decoder and DecoderAttention classes : creating model architecture
- train.py - run this file to run the code in wandb without attention
- train_attention.py - run this file to run the code in wandb without attention
- trainModel.py - contains the code to train the model and computes loss and accuracy

## Folders
- predictions_vanilla - contains a csv file containing the input words, predictions made by the model (without attention) and target words
- predictions_attention - contains a csv file containing the input words, predictions made by the model (with attention) and target words
## Results

### With Attention

####  Validation Accuracy - 35.132 %
####  Test Accuracy - 33.569 %
| Hyperparameter | Values/Usage |
| --- | --- |
| epoch | 10 |
| hidden_size | 512 |
| cell_type | LSTM |
| learning_rate | 1e-3 |
| num_layers | 3 |
| drop_out | 0 |
| embedding_size | 256 |
| batch_size | 128 |
| bidirectional | True |

### Without Attention

####  Validation Accuracy - 35.474 %
#### Test Accuracy - 31.86 %
| Hyperparameter | Values/Usage |
| --- | --- |
| epoch | 10 |
| hidden_size | 512 |
| cell_type | LSTM |
| learning_rate | 1e-3 |
| num_layers | 3 |
| drop_out | 0.3 |
| embedding_size | 256 |
| batch_size | 64 |
| bidirectional | False |

