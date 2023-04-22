from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import pandas as pd


import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATAPATH = 'aksharantar_sampled'
source_lang = 'en'
target_lang = 'hin'
trainPath = os.path.join(DATAPATH,target_lang,target_lang + "_train.csv")
trainData = pd.read_csv(
            trainPath,
            sep=",",
            names=["source", "target"],
        )
SOS_token = 0
EOS_token = 1


# self.validationPath = os.path.join(DATAPATH,self.target_lang,self.target_lang + "_valid.csv")
# self.testPath = os.path.join(DATAPATH,self.target_lang,self.target_lang + "_test.csv")
        
 