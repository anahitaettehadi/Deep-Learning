
from __future__ import print_function

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords 

import numpy as np
import os, glob, cv2, sys, torch, pdb, random
from torch.utils.data import Dataset

import pdb, sys, os, time
import pandas as pd
from tqdm import tqdm

lem = WordNetLemmatizer()

from utils_modified import q

class word2vec_dataset(Dataset):
    def __init__(self, DATA_SOURCE, CONTEXT_SIZE, FRACTION_DATA, SUBSAMPLING, SAMPLING_RATE):

        vocab, word_to_ix, ix_to_word, training_data = self.load_data(DATA_SOURCE, CONTEXT_SIZE, FRACTION_DATA, SUBSAMPLING, SAMPLING_RATE)
        self.vocab = vocab
        self.word_to_ix = word_to_ix
        self.ix_to_word = ix_to_word
        self.data = torch.tensor(training_data, dtype = torch.long)   
        
    def __getitem__(self, index):  
        x = self.data[index, 0]
        y = self.data[index, 1]
        return x, y

    def __len__(self):
        return len(self.data)              
    
    def gather_training_data(self, split_text, word_to_ix, context_size):        
        training_data = []
        all_vocab_indices = list(range(len(word_to_ix)))
        
        #TODO: create pairs of context and center word incides and them to training data
        
        return training_data
            
    def load_data(self, data_source, context_size, fraction_data, subsampling, sampling_rate):

        stop_words = set(stopwords.words('english')) 
        import gensim.downloader as api
        dataset = api.load("text8")
        data = [d for d in dataset][:int(fraction_data*len([d_ for d_ in dataset]))]
        print(f'fraction of data taken: {fraction_data}/1')
        
        sents = []
        print('forming sentences by joining tokenized words...')
        for d in tqdm(data):
            sents.append(' '.join(d))

        sent_list_tokenized = [word_tokenize(s) for s in sents]
        print('len(sent_list_tokenized): ', len(sent_list_tokenized))

        # remove the stopwords
        sent_list_tokenized_filtered = []
        print('lemmatizing and removing stopwords...')
        for s in tqdm(sent_list_tokenized):
            sent_list_tokenized_filtered.append([lem.lemmatize(w, 'v') for w in s if w not in stop_words])
        
        sent_list_tokenized_filtered, vocab, word_to_ix, ix_to_word = self.gather_word_freqs(sent_list_tokenized_filtered, subsampling, sampling_rate)        
        
        training_data = self.gather_training_data(sent_list_tokenized_filtered, word_to_ix, context_size)
        
        return vocab, word_to_ix, ix_to_word, training_data

    def gather_word_freqs(self, split_text, subsampling, sampling_rate): #here split_text is sent_list

        vocab = {}
        ix_to_word = {}
        word_to_ix = {}
        total = 0.0

        print('building vocab...')
        for word_tokens in tqdm(split_text):
            for word in word_tokens: #for every word in the word list(split_text), which might occur multiple times
                if word not in vocab: #only new words allowed
                    vocab[word] = 0
                    ix_to_word[len(word_to_ix)] = word
                    word_to_ix[word] = len(word_to_ix)
                vocab[word] += 1.0 #count of the word stored in a dict
                total += 1.0 #total number of words in the word_list(split_text)
        return split_text, vocab, word_to_ix, ix_to_word