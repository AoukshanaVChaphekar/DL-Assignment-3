import wandb
import torch
from main import trainForConfigs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Default hyperparameters
CONFIG = {
    "wandb_project": "DL Assignment 3-1",
    "wandb_entity": "cs22m019",
    "hidden_size" : 256,
    "source_lang" : 'en',
    "target_lang" : 'hin',
    "cell_type"   : "gru",
    "num_layers" : 2,
    "drop_out"    : 0.2, 
    "embedding_size" : 256,
    "bidirectional" : False,
    "batch_size" : 32,
    "attention" : True,
    "epoch" : 5,
    "device" : device,
    "learning_rate" : 0.001
} 

# Sweep configuration - possible values
sweep_config = {
    'method' : 'bayes', # grid ,random - generates exponential ways,bayesian  efficient way
    'name' : 'attention_bayes_sweep',
    'metric' : {
        'name' : 'validation accuracy',
        'goal' : 'maximize'
    },
    'parameters': {
        'epoch':{
            'values' : [5,10]
        },
        'hidden_size':{
            'values' : [128,256,512]
        },
        'cell_type':{
            'values' : ["lstm","rnn","gru"]
        },
        'learning_rate':{
            'values' : [1e-2,1e-3]
        },
        'num_layers':{
            'values' : [1,2,3]
        },
        'drop_out':{
            'values' : [0.0,0.2,0.3]
        },
        'embedding_size':{
            'values' : [64,128,256,512]
        },
        'batch_size':{
            'values' : [32,64,128]
        },
        'bidirectional':{
            'values' : [True,False]
        }
        
    }
}

    
def train():
    wandb.init(
                # set the wandb project where this run will be logged
                config = sweep_config
    )
    CONFIG["hidden_size"] = wandb.config.hidden_size
    CONFIG["cell_type"] = wandb.config.cell_type
    CONFIG["numLayers"] = wandb.config.num_layers
    CONFIG["drop_out"] = wandb.config.drop_out
    CONFIG["embedding_size"] = wandb.config.embedding_size
    CONFIG["bidirectional"] = False
    CONFIG["batch_size"] = wandb.config.batch_size
    CONFIG["epoch"] = wandb.config.epoch
    CONFIG["learning_rate"] = wandb.config.learning_rate
    CONFIG["attention"] = True
    CONFIG["device"] = device

    
    wandb.run.name = "cell_type_" + str(wandb.config.cell_type) +  "_numLayers_"+ str(wandb.config.num_layers) 
    
    trainForConfigs(CONFIG,add_wandb=True)

sweep_id = wandb.sweep(sweep = sweep_config,project= CONFIG["wandb_project"])  
wandb.agent(sweep_id=sweep_id,function = train,count=50,project = CONFIG["wandb_project"])  