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
        emb_product = torch.mul(context_u_o, center_v_c)
        print(emb_product)
        out_loss = F.logsigmoid(emb_product)

        noise_dist = self.noise_dist            
        samples = self.negative_samples*context_word.shape[0]
        negative_example = torch.multinomial(noise_dist, samples, replacement = True)
        negative_example = negative_example.view(context_word.shape[0], self.negative_samples).to(self.device)

        emb_negative = self.embeddings_context(negative_example)
        emb_product_neg_samples = torch.bmm(emb_negative.neg(), center_v_c.unsqueeze(2))
        noise_loss = F.logsigmoid(emb_product_neg_samples).squeeze(2).sum(1)
        total_loss = (- out_loss - noise_loss).mean()
        return total_loss