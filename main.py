from __future__ import print_function
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys, pdb, os, shutil, pickle
from pprint import pprint 

import torch
import torch.optim as optim
import torch.nn as nn


from model import Word2Vec_neg_sampling
from utils_modified import count_parameters
from datasets import word2vec_dataset
from  config import *
from utils_modified import q, print_nearest_words

import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

if not os.path.exists(MODEL_DIR):
  os.makedirs(MODEL_DIR)

if not os.path.exists(PREPROCESSED_DATA_PATH):
    train_dataset = word2vec_dataset(DATA_SOURCE, CONTEXT_SIZE, FRACTION_DATA, SUBSAMPLING, SAMPLING_RATE)

    if not os.path.exists(PREPROCESSED_DATA_DIR):
        os.makedirs(PREPROCESSED_DATA_DIR)
    print('\ndumping pickle...')
    outfile = open(PREPROCESSED_DATA_PATH,'wb')
    pickle.dump(train_dataset, outfile)
    outfile.close()
    print('pickle dumped\n')

else:
    print('\nloading pickle...')
    infile = open(PREPROCESSED_DATA_PATH,'rb')
    train_dataset = pickle.load(infile)
    infile.close()
    print('pickle loaded\n')

vocab = train_dataset.vocab
word_to_ix = train_dataset.word_to_ix
ix_to_word = train_dataset.ix_to_word

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = not True)
print('len(train_dataset): ', len(train_dataset))
print('len(train_loader): ', len(train_loader))
print('len(vocab): ', len(vocab), '\n')

# make noise distribution to sample negative examples from
word_freqs = np.array(list(vocab.values()))
unigram_dist = word_freqs/sum(word_freqs)
noise_dist = torch.from_numpy(unigram_dist**(0.75)/np.sum(unigram_dist**(0.75)))

losses = []

model = Word2Vec_neg_sampling(EMBEDDING_DIM, len(vocab), DEVICE, noise_dist, NEGATIVE_SAMPLES).to(DEVICE)
print('\nWe have {} Million trainable parameters here in the model'.format(count_parameters(model)))
optimizer = optim.Adam(model.parameters(), lr = LR)

for epoch in tqdm(range(NUM_EPOCHS)):
    print('\n===== EPOCH {}/{} ====='.format(epoch + 1, NUM_EPOCHS))    
    for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
        print('batch# ' + str(batch_idx+1).zfill(len(str(len(train_loader)))) + '/' + str(len(train_loader)), end = '\r')
        
        model.train()

        x_batch           = x_batch.to(DEVICE)
        y_batch           = y_batch.to(DEVICE)
        
        optimizer.zero_grad()
        loss = model(x_batch, y_batch)
        
        loss.backward()
        optimizer.step()    
        losses.append(loss.item())
       
        if batch_idx%DISPLAY_EVERY_N_BATCH == 0 and DISPLAY_BATCH_LOSS:
            print(f'Batch: {batch_idx+1}/{len(train_loader)}, Loss: {loss.item()}')    
            print_nearest_words(model, TEST_WORDS, word_to_ix, ix_to_word, top = 5)        
 

