import numpy as np
import sys
import torch, sys, pdb, os
from model import Word2Vec_neg_sampling
from utils_modified import nearest_word
from config import EMBEDDING_DIM, MODEL_DIR, DEVICE



def q():
    sys.exit()

# define a function to count the total number of trainable parameters
def count_parameters(model): 
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_parameters/1e6 # in terms of millions

def nearest_word(inp, emb, top = 5, debug = False):
    euclidean_dis = np.linalg.norm(inp - emb, axis = 1)    
    emb_ranking = np.argsort(euclidean_dis)
    emb_ranking_distances = euclidean_dis[emb_ranking[:top]]
    
    emb_ranking_top = emb_ranking[:top]
    euclidean_dis_top = euclidean_dis[emb_ranking_top]
    
    if debug:
        print('euclidean_dis: ', euclidean_dis)
        print('emb_ranking: ', emb_ranking)
        print(f'top {top} embeddings are: {emb_ranking[:top]} with respective distances\n {euclidean_dis_top}')
    
    return emb_ranking_top, euclidean_dis_top


def print_nearest_words(model, test_words, word_to_ix, ix_to_word, top = 5):
    
    model.eval()
    emb_matrix = model.embeddings_input.weight.data.cpu()
    
    nearest_words_dict = {}

    print('==============================================')
    for t_w in test_words:
        
        inp_emb = emb_matrix[word_to_ix[t_w], :]  
        emb_ranking_top, _ = nearest_word(inp_emb, emb_matrix, top = top+1)
        print(t_w.ljust(10), ' | ', ', '.join([ix_to_word[i] for i in emb_ranking_top[1:]]))

    return nearest_words_dict

