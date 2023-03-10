import os, torch
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print('DEVICE: ', DEVICE)
DATA_SOURCE           = 'gensim' # or 'toy'
MODEL_ID              = 'gensim' #'toy'# 'gensim'
DISPLAY_BATCH_LOSS    = True
DISPLAY_EVERY_N_BATCH = 1000
SAVE_EVERY_N_EPOCH    = 1
BATCH_SIZE            = 1024*16
NUM_EPOCHS            = 10
CONTEXT_SIZE          = 5
FRACTION_DATA         = 1
SUBSAMPLING           = True
SAMPLING_RATE         = 0.001
LR                    = 0.003
NEGATIVE_SAMPLES      = 20
EMBEDDING_DIM         = 100
TEST_WORDS            = ['india', 'computer', 'gold', 'football', 'cars', 'war', 'apple', 'music', 'helicopter']
TEST_WORDS_VIZ        = ['india', 'asia', 'guitar', 'piano', 'album', 'music', 'war', 'soldiers', 'helicopter']
PREPROCESSED_DATA_DIR  = os.path.join(MODEL_ID, 'preprocessed_data')
PREPROCESSED_DATA_PATH = os.path.join(PREPROCESSED_DATA_DIR, 'preprocessed_data_' + MODEL_ID + '_' + str(FRACTION_DATA) + '.pickle')
SUMMARY_DIR            = os.path.join(MODEL_ID, 'summary') 
MODEL_DIR              = os.path.join(MODEL_ID, 'models')