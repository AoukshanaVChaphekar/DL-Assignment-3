# DL-Assignment-3

The **wandb report** can be found in the following link:


## Usage
### To run code with wandb

- With Attention
```python
python train_attention.py
```
- Without Attention
```python
python train.py
``` 

### To run code without wandb

- With Attention
```python
python main_attention.py
```
- Without Attention
```python
python main.py
``` 


| Hyperparamter | Values/Usage |
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
