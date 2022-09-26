import numpy as np
import os 
from tqdm import tqdm
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
from scipy.stats import spearmanr

from loguru import logger


from model import simcse_unsup_loss, simcse_sup_loss, simcse_sup_loss_with_mask
from model import SimcseModel

from dataset import SquadZenTrainDataset, DureaderTrainDataset
from dataset import TestDataset

from config import (DATA_NAME,
                    EPOCHS,
                    TRAIN_BATCH_SIZE,
                    EVAL_BATCH_SIZE,
                    TRAIN_FILE,
                    TEST_FILE,
                    SAVE_PATH,
                    LR,
                    DEVICE,
                    POOLING,
                    PRETRAIN_MODEL_NAME_OR_PATH,)

if not os.path.exists(f"./logs/{DATA_NAME}"):
    os.makedirs(f"./logs/{DATA_NAME}")
    
writer = SummaryWriter(f"./logs/{DATA_NAME}")

def evaluate(model, dataloader) -> float:
    device = DEVICE
    model.eval()
    sim_tensor = torch.tensor([], device=device)
    label_array = np.array([])
    with torch.no_grad():
        for idx, (qas, doc, score) in enumerate(dataloader):
            
            for k in qas:
                qas[k] = qas[k].to(device)
            for k in doc:
                doc[k] = doc[k].to(device)

            qas_emb = model(**qas)
            doc_emb = model(**doc)

            # concat
            sim = F.cosine_similarity(qas_emb, doc_emb, dim=-1)
            sim_tensor = torch.cat((sim_tensor, sim), dim=0)
            label_array = np.append(label_array, np.array(score))
    # corrcoef
    return spearmanr(label_array, sim_tensor.cpu().numpy()).correlation


def train(model, train_dl, dev_dl, optimizer, loss_fn) -> None:
    device = DEVICE
    global best
    global global_setp
    model.train()
    early_stop_batch = 0

    for batch_idx, (model_input, label, mask) in enumerate(tqdm(train_dl), start=1):
        # move to device
        global_setp += 1
        for k in model_input:
            model_input[k] = model_input[k].to(device)
        if label is not None:
            label = label.to(device)
        if mask is not None:
            mask = mask.to(device)
        
        out = model(**model_input)
        
        loss = loss_fn(out, label, mask)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        writer.add_scalar(f"{DATA_NAME}-Loss/train", loss, global_setp)
        # evaluation
        if batch_idx % int(0.5 * len(train_dl)) == 0:
            logger.info(f'loss: {loss.item():.4f}')
            corrcoef = evaluate(model, dev_dl)
            writer.add_scalar(f"{DATA_NAME}-Test/spear_cof", corrcoef, global_setp)
            model.train()
            if best < corrcoef:
                # early_stop_batch = 0
                best = corrcoef
                torch.save(model.state_dict(), SAVE_PATH)
                logger.info(f"higher corrcoef: {best:.4f} in batch: {batch_idx}, save model")
                continue
            
            
if __name__ == '__main__':
    model_path = PRETRAIN_MODEL_NAME_OR_PATH
    logger.info(f'device: {DEVICE}, pooling: {POOLING}, model path: {model_path}')
    
    if DATA_NAME == "squadzen":
        train_dataset = SquadZenTrainDataset(TRAIN_FILE)
        loss_fn = simcse_sup_loss_with_mask
    elif DATA_NAME == "dureader":
        train_dataset = DureaderTrainDataset(TRAIN_FILE)
        loss_fn = simcse_unsup_loss
    else:
        raise ValueError(f"data name error, {DATA_NAME}")
    
    test_dataset = TestDataset(TEST_FILE)
    
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=TRAIN_BATCH_SIZE,
                                  collate_fn=train_dataset.collate_fn,
                                  shuffle=True)
    
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=EVAL_BATCH_SIZE,
                                 collate_fn=test_dataset.collate_fn,
                                 shuffle=False)
    # load model    
    assert POOLING in ['cls', 'pooler', 'last-avg', 'first-last-avg']
    model = SimcseModel(pretrained_model=model_path, pooling=POOLING)
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    # train
    best = 0
    global_setp = 0
    for epoch in range(EPOCHS):
        logger.info(f'epoch: {epoch}')
        train(model, train_dataloader, test_dataloader, optimizer, loss_fn)
    logger.info(f'train is finished, best model is saved at {SAVE_PATH}')
    
    # eval
    model.load_state_dict(torch.load(SAVE_PATH))
    test_corrcoef = evaluate(model, test_dataloader)
    logger.info(f'test_corrcoef: {test_corrcoef:.4f}')
