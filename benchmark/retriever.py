from typing import List, Union, Tuple, Dict
from sentence_transformers import SentenceTransformer, models
import faiss
import numpy as np
import jsonlines
from tqdm import tqdm
import os 


BERT = "bert-base-chinese"
HFL_ROBERTA = "hfl/chinese-roberta-wwm-ext"

MULTILINUGA = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

CYCLONE_SIMCSE = "cyclone/simcse-chinese-roberta-wwm-ext"
ZEN_SIMCSE = "/home/nvidia/simcse/RAYZ/mrc_simcse_zen"

MODEL_NAME = UER_SIMCSE

DATA_NAME = "/home/nvidia/simcse/benchmark/dureader_robust"
DATA_PATH = "/home/nvidia/simcse/data/dureader_robust-data/preprocessed"


if not os.path.exists(DATA_NAME):
    os.mkdir(DATA_NAME)


class DocRetriever:
    def __init__(self, docs: List[Dict], embedder=None):
        self.docs = []
        self.idx_to_docid = {}
        self.docid_to_text = {}

        for idx, doc in enumerate(docs):
            self.idx_to_docid[idx] = doc.get("doc_id")
            self.docs.append(doc.get("title") + "|" + doc.get("context"))
        

        self.embedder = embedder
        self.faiss = None 
        
        if "/" in MODEL_NAME:
            self.name = MODEL_NAME.split("/")[-1]
        else:
            self.name = MODEL_NAME
        
        
    def set_embedder(self, embedder: SentenceTransformer):
        self.embedder = embedder

    def build_index(self, doc_embs=None, device=0):
        assert self.embedder is not None, "embedder is not set"
        assert self.name is not None, "name is not set"
        
        name = self.name

        d = self.embedder.get_sentence_embedding_dimension()
        self.faiss = faiss.IndexFlatIP(d)
        
        if doc_embs is None:
            if os.path.exists(f"{DATA_NAME}/{name}.npy"):
                print("loading doc embs from file")
                doc_embs = np.load(f"{DATA_NAME}/{name}.npy")
            else:
                print("encoding doc embs")
                doc_embs = self.embedder.encode(self.docs, device=device, show_progress_bar=True)
                np.save(f"{DATA_NAME}/{name}.npy", doc_embs)
        
        self.faiss.add(doc_embs)

    def retrieve(self, qas: Union[List[str], str], topk=1, return_text=False, device=0):
        if isinstance(qas, str):
            qas = [qas]

        assert self.embedder is not None, "embedder is not set"
        assert self.faiss is not None, "faiss is not built"
        assert isinstance(qas, list), "qas should be a list of str"

        print("encoding query embs")
        qas_embs = self.embedder.encode(qas, device=device, show_progress_bar=True)

        # indexs bz * topk
        scores, indexs = self.faiss.search(qas_embs, topk)
        res = {}

        idx2did = np.vectorize(lambda x: self.idx_to_docid[x])

        res['scores'] = scores
        res['docids'] = idx2did(indexs)
        if return_text:
            idx2text = np.vectorize(lambda x: self.docs[x])
            res["text"] = idx2text(indexs)

        return res


def test_doc_retriever(retriever, qas_map, qas_doc, topk=10):
    mrr = 0
    for idx, qd_pair in enumerate(tqdm(qas_doc)):
        qas_id = qd_pair.get("qas_id")
        doc_id = qd_pair.get("doc_id")
        question = qas_map.get(qas_id)

        if question is not None:
            res = retriever.retrieve(question, topk=topk)
            docids = res.get("docids")
            x, rank = np.where(docids == doc_id)
            if len(rank) > 0:
                rr = 1 / (rank[0] + 1)
            else:
                rr = 0
            mrr += rr

    mrr /= len(qas_doc)
    print(MODEL_NAME)
    print(f"mrr@{topk} is {mrr}")

def batch_test_doc_retriever(retriever, qas_map, qas_doc, topk=10):
    
    questions = []
    doc_ids = []
    for idx, qd_pair in enumerate(qas_doc):
        qas_id = qd_pair.get("qas_id")
        doc_id = qd_pair.get("doc_id")
        question = qas_map.get(qas_id)
    
        questions.append(question)
        doc_ids.append(doc_id)
    
    res = retriever.retrieve(questions, topk=topk)
    pred_docids = res.get("docids")
    
    mrr = 0
    for true_id, pre_ids in zip(doc_ids, pred_docids):
        rank, = np.where(pre_ids == true_id)
        if len(rank) > 0:
            rr = 1 / (rank[0] + 1)
        else:
            rr = 0
        mrr += rr
        
    mrr /= len(qas_doc)
    print(DATA_NAME)
    print(MODEL_NAME)
    print(f"mrr@{topk} is {mrr}")
    
    

def build_model(model_name):
    if "sentence-transformers" in model_name:
        return SentenceTransformer(model_name)
    
    word_embedding_model = models.Transformer(model_name, max_seq_length=512)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode="cls")
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    
    return model 
    

if __name__ == "__main__":

    with jsonlines.open(f"{DATA_PATH}/documents.jsonl", 'r') as f:
        docs = list(f)

    with jsonlines.open(f"{DATA_PATH}/questions.jsonl", 'r') as f:
        qas = list(f)

    with jsonlines.open(f"{DATA_PATH}/dev_pairs.jsonl", 'r') as f:
        qas_doc = list(f)

    qas_map = {}
    for q in qas:
        q_id = q.get("qas_id")
        text = q.get("qas")
        qas_map[q_id] = text

    model = build_model(MODEL_NAME)
    retriever = DocRetriever(docs, embedder=model)
    retriever.build_index()

    batch_test_doc_retriever(retriever, qas_map, qas_doc, topk=5)
