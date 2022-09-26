import torch 
# file path
QAS_FILE="/home/nvidia/simcse/data/squad_zen/preprocessed/questions.jsonl"
DOC_FILE="/home/nvidia/simcse/data/squad_zen/preprocessed/documents.jsonl"

TRAIN_FILE="/home/nvidia/simcse/data/squad_zen/preprocessed/train_pairs.jsonl"
TEST_FILE="/home/nvidia/simcse/data/squad_zen/preprocessed/test_score_pairs.jsonl"

SAVE_PATH="/home/nvidia/simcse/mrc-simcse/saved_models/mrc-simcse-zen.pt"

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

PRETRAIN_MODEL_NAME_OR_PATH = ROBERTA