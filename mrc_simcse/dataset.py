import torch
import jsonlines
import torch
from transformers import AutoTokenizer
from config import QAS_FILE, DOC_FILE, MAX_SEQ_LEN, PRETRAIN_MODEL_NAME_OR_PATH, TRAIN_FILE, TEST_FILE
import numpy as np


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

TOKENIZER = AutoTokenizer.from_pretrained(PRETRAIN_MODEL_NAME_OR_PATH)


class SquadZenTrainDataset(torch.utils.data.Dataset):
    def __init__(self, pair_file, ):
        super().__init__()

        with jsonlines.open(pair_file, 'r') as f:
            self.pair = list(f)

        self.pos_pairs = set()
        for p in self.pair:
            pos = f"{p['qas_id']}|{p['doc_id']}"
            self.pos_pairs.add(pos)

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

    def is_positive_key(self, q_id, d_id):
        key = f"{q_id}|{d_id}"
        return key in self.pos_pairs

    def collate_fn(self, batch, tokenizer=TOKENIZER):
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

        mask = np.zeros((len(label), len(label)), dtype=bool)

        for i in range(0, len(label), 3):
            for j in range(0, len(label)):
                if j % 3 == 0:
                    continue
                mask[i][j] = self.is_positive_key(label[i], label[j])
                mask[j][i] = mask[i][j]
        mask = torch.tensor(mask, dtype=torch.bool)
        label = torch.tensor(label, dtype=torch.long)
        return encoded, label, mask


class DureaderTrainDataset(torch.utils.data.Dataset):
    def __init__(self, pair_file, ):
        super().__init__()

        with jsonlines.open(pair_file, 'r') as f:
            self.pair = list(f)

        self.db = QAS_DOC_DB

    def __getitem__(self, index):
        pair = self.pair[index]
        qas_id = pair['qas_id']
        doc_id = pair['doc_id']

        text = [self.db.qas[qas_id], self.db.doc[doc_id]]
        label = [qas_id, doc_id]

        return text, label

    def __len__(self):
        return len(self.pair)

    def collate_fn(self, batch, tokenizer=TOKENIZER):
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

        return encoded, label, None


class DureaderRetrievalDataset(torch.utils.data.Dataset):
    def __init__(self, pair_file, ):
        super().__init__()

        with jsonlines.open(pair_file, 'r') as f:
            self.data = list(f)

    def __getitem__(self, index):
        sample = self.data[index]
        return list(sample.values())

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch, tokenizer=TOKENIZER):
        text = []

        for t in batch:
            text.extend(t)

        encoded = tokenizer(text,
                            padding="longest",
                            truncation=True,
                            max_length=MAX_SEQ_LEN,
                            return_tensors="pt")

        return encoded, None, None


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

    def collate_fn(self, batch, tokenizer=TOKENIZER):
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


if __name__ == "__main__":

    # train_dataset = DureaderTrainDataset(TRAIN_FILE)
    # batch = []
    # for t in train_dataset:
    #     batch.append(t)
    #     if len(batch) == 3:
    #         break
    # encoded, label, mask = train_dataset.collate_fn(batch)
    # print(encoded)
    # print(label)
    # print(mask)

    test_dataset = TestDataset(TEST_FILE)
    batch = []
    for i in range(3):
        batch.append(test_dataset[i])
    # print(batch)
    qas, doc, rel = test_dataset.collate_fn(batch)
    print(qas)
    print(doc)
    print(rel)
