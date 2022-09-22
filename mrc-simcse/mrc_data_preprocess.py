import pandas as pd
import jsonlines 
import json 
import numpy as np 
from sklearn.utils import murmurhash3_32
import random 
import pandas as pd 
    
SQUAD_ZEN="/home/nvidia/simcse/data/squad_zen/"


def read_json_file(file_name):
    with open(file_name, 'r', encoding="utf-8") as f:
        data = json.load(f)
    return data 


def write_jsonlines(data, filename):
    with jsonlines.open(filename, 'w') as f:
        f.write_all(data)


def squad_to_samples(squad_data):
    """
    squad_data:
    {
        "data":
        [
            {
                "title": ""
                "paragraphs": 
                [
                    {
                        "context": ""
                        "qas" : 
                        [
                            {
                                "answers":
                                [
                                    {
                                        "answer_start": 58,
                                        "text": "丹佛野马"
                                    },
                                ]
                                "question": "",
                                "is_impossible":bool,
                            }
                        ]
                    }
                ]
            }
        ]
    }
    
    res:
    [
        {
            "title",
            "question",
            "context",
            "ans":[]
        }
    ]
    """
    data = squad_data.get('data')
    pairs = []
    docs = []
    qass = []
    
    
    for document in data:
        title = document.get("title")
        p_id = murmurhash3_32(title)
        
        paragraphs = document.get("paragraphs")
        for paragraph in paragraphs:
            context = paragraph.get("context")
            questions = paragraph.get("qas")
            
            doc_id = murmurhash3_32(title + context)
            docs.append({ "doc_id":doc_id, "title": title, "context":context, "p_id":p_id})
            
            for question in questions:
                if question.get("is_impossible"):
                    continue
                
                question_text = question.get("question")
                answers =  question.get("answers")
                
                qas_id = murmurhash3_32(question_text)
                qass.append({"qas_id": qas_id, "qas": question_text, "ans":answers, "p_id":p_id,})
                pairs.append({"qas_id":qas_id, "doc_id":doc_id, "p_id":p_id})
                
    return qass, docs, pairs
                    
                    
def convert_to_jsonline():
    
    # read original data 
    train_data = read_json_file(SQUAD_ZEN + "train-zen-v1.0.json")
    dev_data = read_json_file(SQUAD_ZEN + "dev-zen-v1.0.json")
    qas, docs, train_pairs = squad_to_samples(train_data)
    dev_qas, dev_docs, dev_pairs = squad_to_samples(dev_data)
    
    qas.extend(dev_qas)
    docs.extend(dev_docs)
    
    # sample hard_negative data 
    train_pairs = resample_hard_negative(train_pairs)
    dev_pairs = resample_hard_negative(dev_pairs)
    
    # shuffle pairs
    random.shuffle(train_pairs)
    random.shuffle(dev_pairs)
    
    test_pairs = sample_eval_pairs(dev_pairs)
    
    # save to file
    write_jsonlines(qas,SQUAD_ZEN + "preprocessed/questions.jsonl")
    write_jsonlines(docs, SQUAD_ZEN + "preprocessed/documents.jsonl")
    write_jsonlines(dev_pairs, SQUAD_ZEN + "preprocessed/dev_paris.jsonl")
    write_jsonlines(train_pairs, SQUAD_ZEN + "preprocessed/train_paris.jsonl")
    write_jsonlines(test_pairs, SQUAD_ZEN + "preprocessed/test_score_pairs.jsonl")
    
    

def resample_hard_negative(pairs):
    df = pd.DataFrame(pairs)
    
    # sample p_id different doc_id
    docs = df[["doc_id", "p_id"]].groupby("p_id").aggregate(lambda x: list(set(list(x))))
    df = pd.merge(df, docs, left_on="p_id", right_on="p_id")
    
    # random select hard_negative context
    def select_hard_negative(curr_id, id_set):
        if len(id_set) <= 1:
            return None 
        hn_id = random.sample(id_set,k=1)[0]
        while curr_id == hn_id:
            hn_id = random.sample(id_set,k=1)[0]
        return hn_id 
    
    df['hn_doc_id'] = df.apply(lambda x: select_hard_negative(x['doc_id_x'], x["doc_id_y"]), axis=1)
    df = df.rename(columns = {"doc_id_x":"doc_id"})
    
    df = df[["qas_id", "doc_id", "hn_doc_id", "p_id"]]
    df.dropna(inplace=True)
    
    data = df.to_dict(orient="records")
    return data


def sample_eval_pairs(pairs):
    df = pd.DataFrame(pairs)
    docs = df[["doc_id", "p_id"]].groupby("p_id").aggregate(lambda x: list(set(list(x))))
    df = pd.merge(df, docs, left_on="p_id", right_on="p_id")
    pos_pairs = []
    neg_pairs = []
    hn_neg_pairs = []
    all_doc_ids = set(df["doc_id_x"].tolist())
    for idx, row in df.iterrows():
        qas_id = row['qas_id']
        pos_id = row['doc_id_x']
        doc_ids = row['doc_id_y']
        for doc_id in doc_ids:
            if pos_id == doc_id:
                pos_pairs.append({"qas_id": qas_id, "doc_id":doc_id, "rel":3})
            else:
                hn_neg_pairs.append({"qas_id": qas_id, "doc_id":doc_id, "rel":1})
        
        neg_ids = random.sample(all_doc_ids, k = len(doc_ids) * 2)
        for neg_id in neg_ids:
            if neg_id not in doc_ids:
                neg_pairs.append({"qas_id": qas_id, "doc_id":neg_id, "rel":0})
    
    num = len(pos_pairs)
    res = []
    res.extend(random.sample(pos_pairs, k = num))
    res.extend(random.sample(neg_pairs, k =  10 * num))
    res.extend(random.sample(hn_neg_pairs, k = 1 * num))
    random.shuffle(res)
    
    return res 
    
    
if __name__=="__main__":
    convert_to_jsonline()
    
    