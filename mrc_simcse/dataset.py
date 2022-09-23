import torch
import jsonlines
import torch
from transformers import AutoTokenizer
from config import QAS_FILE, DOC_FILE, MAX_SEQ_LEN, PRETRAIN_MODEL_NAME_OR_PATH


class QasDocData:
    def __init__(self, qas_file=QAS_FILE, doc_file=DOC_FILE):
        self.load_doc(doc_file)
        self.load_qas(qas_file)

    def load_qas(self, qas_file):
        with jsonlines.open(qas_file, 'r') as f:
            data = list(f)
        self.qas = {d['qas_id']: d['qas'] for d in data}

    def load_doc(self, doc_file):
        with jsonlines.open(doc_file, 'r') as f:
            data = list(f)
        self.doc = {d['doc_id']: d['title'] + "|" + d['context'] for d in data}


QAS_DOC_DB = QasDocData(qas_file=QAS_FILE, doc_file=DOC_FILE)


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, pair_file, ):
        super().__init__()

        with jsonlines.open(pair_file, 'r') as f:
            self.pair = list(f)

        self.db = QAS_DOC_DB

    def __getitem__(self, index):
        pair = self.pair[index]
        qas_id = pair['qas_id']
        doc_id = pair['doc_id']
        hn_doc_id = pair['hn_doc_id']

        text = [self.db.qas[qas_id], self.db.doc[doc_id], self.db.doc[hn_doc_id]]
        label = [qas_id, doc_id, hn_doc_id]

        return text, label

    def __len__(self):
        return len(self.pair)


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, pair_file):
        super().__init__()

        with jsonlines.open(pair_file, 'r') as f:
            self.pair = list(f)

        self.db = QAS_DOC_DB

    def __getitem__(self, index):
        pair = self.pair[index]
        qas_id = pair['qas_id']
        doc_id = pair['doc_id']
        score = pair['rel']
        return self.db.qas[qas_id], self.db.doc[doc_id], score

    def __len__(self):
        return len(self.pair)

TOKENIZER = AutoTokenizer.from_pretrained(PRETRAIN_MODEL_NAME_OR_PATH)


def train_collate_fn(batch, tokenizer=TOKENIZER):
    text = []
    label = []

    for t, l in batch:
        text.extend(t)
        label.extend(l)

    encoded = tokenizer(text,
                        padding="longest",
                        truncation=True,
                        max_length=MAX_SEQ_LEN,
                        return_tensors="pt")

    label = torch.tensor(label, dtype=torch.long)

    return encoded, label


def test_collate_fn(batch, tokenizer=TOKENIZER):
    qastions = []
    documents = []
    scores = []
    for qas, doc, score in batch:
        qastions.append(qas)
        documents.append(doc)
        scores.append(score)

    questions = tokenizer(qastions,
                          padding="longest",
                          truncation=True,
                          max_length=MAX_SEQ_LEN,
                          return_tensors="pt")

    documents = tokenizer(documents,
                          padding="longest",
                          truncation=True,
                          max_length=MAX_SEQ_LEN,
                          return_tensors="pt")

    return questions, documents, scores


if __name__=="__main__":
    train = TrainDataset("/home/nvidia/simcse/data/squad_zen/preprocessed/train_paris.jsonl")
    test =  TestDataset("/home/nvidia/simcse/data/squad_zen/preprocessed/test_score_pairs.jsonl")
    for t in train:
        print(t)
        break