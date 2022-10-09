import torch 
# file path
DATA_NAME = "dureader_retrieval"

if DATA_NAME == "squadzen":
    prefix = "/home/nvidia/simcse/data/squad_zen/preprocessed/"
elif DATA_NAME == "dureader":
    prefix = "/home/nvidia/simcse/data/dureader_robust-data/preprocessed/"
elif DATA_NAME == "dureader_retrieval":
    prefix = "/home/nvidia/simcse/data/dureader_retrieval/simcse_preprocessed/"
    
QAS_FILE=f"{prefix}questions.jsonl"
DOC_FILE=f"{prefix}documents.jsonl"
TRAIN_FILE=f"{prefix}train_pairs.jsonl"
TEST_FILE=f"{prefix}test_score_pairs.jsonl"



# training configuration
EPOCHS = 3
SAMPLES = 10000
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
LR = 1e-5
DROPOUT = 0.3
MAX_SEQ_LEN = 512
POOLING = 'cls'   # choose in ['cls', 'pooler', 'first-last-avg', 'last-avg']
DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# pretrain model path
BERT = 'bert-base-chinese'
BERT_WWM_EXT = 'hfl/chinese-bert-wwm-ext'
ROBERTA = 'hfl/chinese-roberta-wwm-ext'
ZEN_SIMCSE = "/home/nvidia/simcse/RAYZ/mrc_simcse_zen"
ZEN_MULTI_SIMCSE = "/home/nvidia/simcse/RAYZ/squadzen-multi-simcse"

PRETRAIN_MODEL_NAME_OR_PATH = ROBERTA

start_ckp = PRETRAIN_MODEL_NAME_OR_PATH.split("/")[-1]
SAVE_PATH=f"/home/nvidia/simcse/mrc_simcse/saved_models/{DATA_NAME}-{start_ckp}"