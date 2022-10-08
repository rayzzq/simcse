from model import SimcseModel
from transformers import AutoTokenizer
from config import PRETRAIN_MODEL_NAME_OR_PATH, POOLING, SAVE_PATH


if __name__=="__main__":
    model_name = SAVE_PATH.split("/")[-1].split(".")[0]
    
    model = SimcseModel(PRETRAIN_MODEL_NAME_OR_PATH, POOLING)
    import torch 
    model.load_state_dict(torch.load(SAVE_PATH))
    model.bert.save_pretrained(f"/home/nvidia/simcse/RAYZ/{model_name}")

    tokenizer = AutoTokenizer.from_pretrained(PRETRAIN_MODEL_NAME_OR_PATH)
    tokenizer.save_pretrained(f"/home/nvidia/simcse/RAYZ/{model_name}")