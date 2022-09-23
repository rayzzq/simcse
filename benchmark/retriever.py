
from typing import List, Union, Tuple, Dict
from sentence_transformers import SentenceTransformer, models
import faiss
import numpy as np
import jsonlines
from tqdm import tqdm

BERT = "bert-base-chinese"
HFL_ROBERTA = "hfl/chinese-roberta-wwm-ext"
SIMCSE_ZEN_MODEL = "/home/nvidia/simcse/RAYZ/mrc_simcse_zen"
MULTILINUGA = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

MODEL_NAME = HFL_ROBERTA
class DocRetriever:
    def __init__(self, docs: List[Dict], embedder=None):
        self.docs = docs
        self.idx_to_docid = {}
        self.docid_to_text = {}

        for idx, doc in enumerate(docs):
            self.idx_to_docid[idx] = doc.get("doc_id")
            self.docid_to_text[doc.get("doc_id")] = doc.get("title") + "|" + doc.get("context")

        self.embedder = embedder
        self.faiss = None 
        
    def set_embedder(self, embedder: SentenceTransformer):
        self.embedder = embedder

    def build_index(self, doc_embs=None):
        assert self.embedder is not None, "embedder is not set"
        d = self.embedder.get_sentence_embedding_dimension()
        self.faiss = faiss.IndexFlatIP(d)

        if doc_embs is None:
            docs = list(self.docid_to_text.values())
            doc_embs = self.embedder.encode(docs, device=0)
            
            if "/" in MODEL_NAME:
                name = MODEL_NAME.split("/")[-1]
            else:
                name = MODEL_NAME
            np.save(f"./doc_embedding/{name}.npy", doc_embs)
        
        self.faiss.add(doc_embs)

    def load_index(self):
        doc_embs = np.load("doc_embeddings.npy")
        self.faiss.add(doc_embs)

    def retrieve(self, qas: Union[List[str], str], topk=1, return_text=False):
        if isinstance(qas, str):
            qas = [qas]

        assert self.embedder is not None, "embedder is not set"
        assert self.faiss is not None, "faiss is not built"
        assert isinstance(qas, list), "qas should be a list of str"

        qas_embs = self.embedder.encode(qas)

        # indexs bz * topk
        scores, indexs = self.faiss.search(qas_embs, topk)
        res = {}

        idx2did = np.vectorize(lambda x: self.idx_to_docid[x])
        did2text = np.vectorize(lambda x: self.docid_to_text[x])

        res['scores'] = scores
        res['docids'] = idx2did(indexs)
        if return_text:
            res["text"] = did2text(res['docids'])

        return res


def test_doc_retriever(retriever, qas_map, qas_doc, topk=10):
    mrr = 0
    for qd_pair in tqdm(qas_doc):
        qas_id = qd_pair.get("qas_id")
        doc_id = qd_pair.get("doc_id")
        question = qas_map.get(qas_id)

        if question is not None:
            res = retriever.retrieve(question, topk=topk)
            docids = res.get("docids")
            x, rank = np.where(docids == doc_id)
            if len(x) > 0:
                rr = 1 / (x[0] + 1)
            else:
                rr = 0
            mrr += rr

    mrr /= len(qas_doc)
    print(f"mrr@{topk} is {mrr}")

def build_model(model_name):
    if "sentence-transformers" in model_name:
        return SentenceTransformer(model_name)
    
    word_embedding_model = models.Transformer(model_name, max_seq_length=512)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode="cls")
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    
    return model 
    

if __name__ == "__main__":

    with jsonlines.open("/home/nvidia/simcse/data/squad_zen/preprocessed/documents.jsonl", 'r') as f:
        docs = list(f)

    with jsonlines.open("/home/nvidia/simcse/data/squad_zen/preprocessed/questions.jsonl", 'r') as f:
        qas = list(f)

    with jsonlines.open("/home/nvidia/simcse/data/squad_zen/preprocessed/dev_paris.jsonl", 'r') as f:
        qas_doc = list(f)

    qas_map = {}
    for q in qas:
        q_id = q.get("qas_id")
        text = q.get("qas")
        qas_map[q_id] = text

    model = build_model(MODEL_NAME)
    retriever = DocRetriever(docs, embedder=model)
    retriever.build_index()

    test_doc_retriever(retriever, qas_map, qas_doc, topk=10)
