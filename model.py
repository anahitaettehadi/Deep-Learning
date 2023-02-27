from __future__ import print_function
import torch, sys, pdb
import torch.nn as nn
import torch.nn.functional as F

class Word2Vec_neg_sampling(nn.Module):

    def __init__(self, embedding_size, vocab_size, device, noise_dist = None, negative_samples = 10):
        super(Word2Vec_neg_sampling, self).__init__()

        self.embeddings_input = nn.Embedding(vocab_size, embedding_size)
        self.embeddings_context = nn.Embedding(vocab_size, embedding_size)
        self.vocab_size = vocab_size
        self.negative_samples = negative_samples
        self.device = device
        self.noise_dist = noise_dist
        self.embeddings_input.weight.data.uniform_(-1,1)
        self.embeddings_context.weight.data.uniform_(-1,1)

    def forward(self, input_word, context_word):
        
        #TODO sample self.negative_samples*context_word.shape[0] from noise_dist using torch.multinomial and then compute the loss function using F.logsigmoid
        #...
        #...
        #Return the loss 
        print('test')