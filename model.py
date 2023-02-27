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
        center_v_c = self.embeddings_input(input_word)
        context_u_o = self.embeddings_context(context_word)
        batch_size, embed_size = center_v_c.shape
        
        center_v_c = center_v_c.view(batch_size, embed_size, 1)   # batch of column vectors
        context_u_o = context_u_o.view(batch_size, 1, embed_size) # batch of row vectors
        out_loss = F.logsigmoid(torch.bmm(context_u_o, center_v_c)).squeeze()

        neg_samples = batch_size * self.negative_samples
        noise_dist = self.noise_dist

        noise_words = torch.multinomial(noise_dist, neg_samples, replacement = True)
        noise_words = noise_words.to(self.device)
        noise_u_k = self.embeddings_context(noise_words)
        noise_u_k = noise_u_k.view(batch_size, self.negative_samples, embed_size)

        noise_loss = torch.bmm(noise_u_k.neg(), center_v_c) 
        noise_loss = F.logsigmoid(noise_loss).squeeze().sum(1)
        
        return - (out_loss + noise_loss).mean()
