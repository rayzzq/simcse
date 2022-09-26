import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, BertConfig, BertModel
from config import *
from torch.nn import functional as F


def simcse_unsup_loss(sent_emb, label, mask, temp = 0.05):
    device = sent_emb.device
    y_true = torch.arange(sent_emb.shape[0], device=device)
    y_true = (y_true - y_true % 2 * 2) + 1
    
    sim = F.cosine_similarity(sent_emb.unsqueeze(1), sent_emb.unsqueeze(0), dim=-1)
    sim = sim / temp
    sim = sim - 1e12 * torch.eye(sim.shape[0], device=device)
    loss = F.cross_entropy(sim, y_true)
    
    return loss

def simcse_sup_loss(sent_emb, label, mask, temp=0.05):
    """
    sent_emb [bz * 3, 768] torch.float
    label[bz * 3] torch.long
    """
    device = sent_emb.device

    y_true = torch.arange(sent_emb.shape[0], device=device)
    use_row = torch.where((y_true + 1) % 3 != 0)[0]
    y_true = (use_row - use_row % 3 * 2) + 1

    # [bz * 3, bz * 3]
    sim = F.cosine_similarity(sent_emb.unsqueeze(1), sent_emb.unsqueeze(0), dim=-1)
    sim = sim / temp
    sim = torch.where(label.unsqueeze(1) == label.unsqueeze(0), -1e12, sim)
    sim = torch.index_select(sim, 0, use_row)

    loss = F.cross_entropy(sim, y_true)

    return loss

def simcse_sup_loss_with_mask(sent_emb, label, mask, temp=0.05):
    """
    sent_emb [bz * 3, 768] torch.float
    label[bz * 3] torch.long
    """
    # [bz * 3, bz * 3]
    sim = F.cosine_similarity(sent_emb.unsqueeze(1), sent_emb.unsqueeze(0), dim=-1)
    sim = sim / temp
    sim = torch.where(label.unsqueeze(1) == label.unsqueeze(0), -1e12, sim)

    sim = F.log_softmax(sim, dim=-1)
    
    # select true position (positive pairs)
    sim = torch.where(mask, sim, 0)
    
    # sum all postive pairs
    normalizer = torch.sum(mask, dim=-1)
    sim_sum = torch.sum(sim, dim=-1)
    
    # normalize
    sim_sum = torch.masked_select(sim_sum, normalizer.bool())
    normalizer = torch.masked_select(normalizer, normalizer.bool())
    
    # reduction use mean, negative likelihood
    loss = torch.mean(sim_sum / normalizer)
    loss = -loss
    
    return loss

class SimcseModel(nn.Module):
    def __init__(self, pretrained_model, pooling):
        super(SimcseModel, self).__init__()
        config = BertConfig.from_pretrained(pretrained_model)
        config.attention_probs_dropout_prob = DROPOUT   # 修改config的dropout系数
        config.hidden_dropout_prob = DROPOUT
        self.bert = BertModel.from_pretrained(pretrained_model, config=config)
        self.pooling = pooling

    def forward(self, *args, **kwargs):

        out = self.bert(*args, **kwargs, output_hidden_states=True)

        if self.pooling == 'cls':
            return out.last_hidden_state[:, 0]  # [batch, 768]

        if self.pooling == 'pooler':
            return out.pooler_output            # [batch, 768]

        if self.pooling == 'last-avg':
            last = out.last_hidden_state.transpose(1, 2)    # [batch, 768, seqlen]
            return torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)       # [batch, 768]

        if self.pooling == 'first-last-avg':
            first = out.hidden_states[1].transpose(1, 2)    # [batch, 768, seqlen]
            last = out.hidden_states[-1].transpose(1, 2)    # [batch, 768, seqlen]
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)   # [batch, 768]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)     # [batch, 2, 768]
            return torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)     # [batch, 768]
