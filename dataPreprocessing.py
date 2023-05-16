import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2
import pathlib
import torch
from torch.utils.data import DataLoader

class DataProcessing():

    # Constructor for DataProcessing
    def __init__(self, DATAPATH, source_lang, target_lang,device,config):
        
        # Source language
        self.source_lang = source_lang

        # Target language
        self.target_lang = target_lang
        
        # Start of Word and its Integer representation
        self.SOW = '>'
        self.SOW_char2int = 0

        # End of Word and its Integer representation
        self.EOW = '<'
        self.EOW_char2int = 1

        # Padding and its Integer representation
        self.PAD = '.'
        self.PAD_char2int = 2

        # Unknown and its Integer representation
        self.UNK = '?'
        self.UNK_char2int = 3 

        self.device = device
        self.batch_size = config["batch_size"]
        
        # Get path of train,validation and test data
        self.trainPath = os.path.join(DATAPATH,self.target_lang,self.target_lang + "_train.csv")
        self.validationPath = os.path.join(DATAPATH,self.target_lang,self.target_lang + "_valid.csv")
        self.testPath = os.path.join(DATAPATH,self.target_lang,self.target_lang + "_test.csv")
        
        # Load train data and set the column names - [source,target]
        self.train = pd.read_csv(
            self.trainPath,
            sep=",",
            names=["source", "target"],
        )

        # Load validation data and set the column names - [source,target]
        self.val = pd.read_csv(
            self.validationPath,
            sep=",",
            names=["source", "target"],
        )

        # Load test data and set the column names - [source,target]
        self.test = pd.read_csv(
            self.testPath,
            sep=",",
            names=["source", "target"],
        )
        
        # Creates train data
        self.train_data = self.preprocess(self.train["source"].to_list(), self.train["target"].to_list())

        # Store encoder input,decoder input,decoder target,source vocabulary,target vocabulary
        self.trainEncodeInput,self.trainDecoderInput,self.trainDecoderTarget,self.source_vocab,self.target_vocab = self.train_data
        
        self.source_char2int,self.source_int2char = self.source_vocab
        self.target_char2int,self.target_int2char = self.target_vocab
        self.max_length = self.getMaxLength()
       
    def encode(self, source_words, target_words, source_chars, target_chars, source_char2int = None, target_char2int = None):
        '''
        Input - 1.source_words - list of all source words
                2.target_words - list of all target words
                3.source_chars - sorted list of all characters in source language
                4.target_chars - sourted list of all characters in target langauge
                5.source_char2int - Dictionary mappig of charcater to integer for source words
                6.target_char2int - Dictionary mappig of charcater to integer for target words
        ''' 

        # Generate source and target vocab pairs containing dictionary mapping of character to integer and integer to character for source and target words
        source_vocab, target_vocab = None, None
        if source_char2int == None and target_char2int == None:

            source_char2int = dict([(char, i + 4) for i, char in enumerate(source_chars)])
            target_char2int = dict([(char, i + 4) for i, char in enumerate(target_chars)])

            source_int2char = dict([(i + 4, char) for i, char in enumerate(source_chars)])
            target_int2char = dict([(i + 4, char) for i, char in enumerate(target_chars)])

            # Add SOW to dictionaries
            source_char2int[self.SOW] = self.SOW_char2int
            source_int2char[self.SOW_char2int] = self.SOW
            target_char2int[self.SOW] = self.SOW_char2int
            target_int2char[self.SOW_char2int] = self.SOW

            # Add EOW to dictionaries
            source_char2int[self.EOW] = self.EOW_char2int
            source_int2char[self.EOW_char2int] = self.EOW
            target_char2int[self.EOW] = self.EOW_char2int
            target_int2char[self.EOW_char2int] = self.EOW

            # Add PAD to dictionaries
            source_char2int[self.PAD] = self.PAD_char2int
            source_int2char[self.PAD_char2int] = self.PAD
            target_char2int[self.PAD] = self.PAD_char2int
            target_int2char[self.PAD_char2int] = self.PAD

            # Add UNK to dictionaries
            source_char2int[self.UNK] = self.UNK_char2int
            source_int2char[self.UNK_char2int] = self.UNK
            target_char2int[self.UNK] = self.UNK_char2int
            target_int2char[self.UNK_char2int] = self.UNK

            source_vocab = (source_char2int,source_int2char)
            target_vocab = (target_char2int,target_int2char)
        
        
        self.encoder_input_data = np.zeros((len(source_words), self.max_source_length,self.num_encoder_tokens), dtype="float32")
        self.decoder_input_data = np.zeros((len(source_words), self.max_target_length,self.num_decoder_tokens), dtype="float32")
        self.decoder_target_data = np.zeros((len(source_words), self.max_target_length,self.num_decoder_tokens), dtype="float32")
            
        
        if source_vocab != None and target_vocab != None:
            return (
                    self.encoder_input_data,
                    self.decoder_input_data,
                    self.decoder_target_data,
                    source_vocab,
                    target_vocab,
                )
        
        # Source and TargetVocab were not created in the function. 
        # This implies sourceCharToInt and targetCharToInt were not None. Hence the vocab info is already present and we don't return the two tuples.
        else:
            return self.encoder_input_data, self.decoder_input_data, self.decoder_target_data

    def preprocess(self, source , target):
       
        # Creating list of words used in source language and converting them into string
        self.source_words = []
        for src in source:
            self.source_words.append(str(src))
        
        # Creating list of words used in target language and converting them into string
        self.target_words = []
        for trg in target:
            self.target_words.append(str(trg))
        
        # Set used to store characters used in source language
        source_chars = set()
        
        # Set used to store characters used in target language
        target_chars = set()

        # Populate source_chars and target_chars
        for src, tgt in zip(self.source_words, self.target_words):
            for char in src:
                source_chars.add(char)

            for char in tgt:
                target_chars.add(char)

        
        # Total number of training samples
        self.number_of_train_samples = len(self.source_words)
        
        
        # Sort the characters used in source and target language       
        source_chars = sorted(list(source_chars))
        target_chars = sorted(list(target_chars))

        # Number of unique characters in source language + Start_of_word,end_of_word,padding and unknown token
        self.num_encoder_tokens = len(source_chars) + 4
        
        # Number of unique characters in target language + Start_of_word,end_of_word,padding and unknown token
        self.num_decoder_tokens = len(target_chars) + 4

        # Length of maximum word in source_words
        self.max_source_length = max([len(txt) for txt in self.source_words])

        # Length of maximum word in target_words
        self.max_target_length = max([len(txt) for txt in self.target_words])

        return self.encode(self.source_words, self.target_words, source_chars, target_chars)

    # Returns list of integer mapped to character
    def indexesFromWord(self,lang,word):
        indexes = []
        if lang == "source":
            for char in word:
                # If character is in dictionary,add it to the list ,else add unknown token
                if char in self.source_char2int:
                    indexes.append(self.source_char2int[char])
                else:
                    indexes.append(self.UNK_char2int)
        if lang == "target":
            for char in word:
                # If character is in dictionary,add it to the list ,else add unknown token
                if char in self.target_char2int:
                    indexes.append(self.target_char2int[char])
                else:
                    indexes.append(self.UNK_char2int)

        return indexes

    # Create tensor for word
    def tensorFromWord(self,lang, word):
        # Gets list of integer mapped to character
        indexes = self.indexesFromWord(lang, word)
        
        # Append EOW2Int 
        indexes.append(self.EOW_char2int)
        max_length = self.max_length
        
        # Add padding
        len_padding = max_length - len(indexes) + 1
        
        indexes.extend([self.PAD_char2int for i in range(len_padding)])
        
        return torch.tensor(indexes, dtype = torch.long, device = self.device).view(-1, 1)

    # Create tensor from pair 
    def tensorsFromPair(self,pairs):
        # Get the source and target word for a given pair and generate tensors for them
        source_tensor = self.tensorFromWord("source", pairs[0])
        target_tensor = self.tensorFromWord("target", pairs[1])

        return (source_tensor, target_tensor)
    
    # Create list of pairs conataining source_words and target_words for train data
    def createTrainPairs(self):
        pairs  = []
        source_words = self.source_words
        target_words = self.target_words

        for source_word,target_word in zip(source_words,target_words):
            pairs.append((source_word,target_word))
        return pairs
    
    # Create list of pairs conataining source_words and target_words for validation data
    def createValidationData(self):
        pairs = []
        source_words = []
        target_words = []
        for word in self.val["source"].to_list():
            source_words.append(word)

        for word in self.val["target"].to_list():
            target_words.append(word)
        
        for source_word,target_word in zip(source_words,target_words):
            pairs.append((source_word,target_word))

        return pairs
    
    # Create list of pairs conataining source_words and target_words for test data
    def createTestData(self):
        pairs = []
        source_words = []
        target_words = []
        for word in self.test["source"].to_list():
            source_words.append(word)

        for word in self.test["target"].to_list():
            target_words.append(word)
        
        for source_word,target_word in zip(source_words,target_words):
            pairs.append((source_word,target_word))

        return pairs
    
    # Returns the maximum length of word from source_lang and target_lang
    def getMaxLength(self):
        
        source_words = []
        target_words = []
        for word in self.val["source"].to_list():
            source_words.append(word)

        for word in self.val["target"].to_list():
            target_words.append(word)
        
        val_max_source_length = max([len(txt) for txt in source_words])
        val_max_target_length = max([len(txt) for txt in target_words])
        
        source_words = []
        target_words = []
        for word in self.test["source"].to_list():
            source_words.append(word)

        for word in self.test["target"].to_list():
            target_words.append(word)
        
        test_max_source_length = max([len(txt) for txt in source_words])
        test_max_target_length = max([len(txt) for txt in target_words])

        self.maxSourceLength = max([self.max_source_length,val_max_source_length,test_max_source_length])
        self.maxTargetLength = max([self.max_target_length,val_max_target_length,test_max_target_length])

        return max(self.maxSourceLength,self.maxTargetLength)
    
    # Returns loader for validation data
    def getValLoader(self):
        
        validationPairs = self.createValidationData()
        validation_pairs = []
        for pair in validationPairs:
            validation_pairs.append(self.tensorsFromPair(pair))
        
        val_Loader = DataLoader(validation_pairs,batch_size = self.batch_size,shuffle = True)
        return val_Loader
    
    # Returs loader for test data
    def getTestLoader(self):
        testPairs = self.createTestData()

        test_pairs = []
        for pair in testPairs:
            test_pairs.append(self.tensorsFromPair(pair))
        
        test_Loader = DataLoader(test_pairs,batch_size = self.batch_size,shuffle = True)

        return test_Loader
    
    