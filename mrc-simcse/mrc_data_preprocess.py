import pandas as pd
import jsonlines 
import json 
import numpy as np 
from sklearn.utils import murmurhash3_32
import random 
import pandas as pd 
    
SQUAD_ZEN="/home/nvidia/finetune-simcse/data/squad_zen/"


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
                qass.append({"qas_id": qas_id, "qas": question, "ans":answers, "p_id":p_id,})
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
    
    # save to file
    write_jsonlines(qas,SQUAD_ZEN + "preprocessd/questions.jsonl")
    write_jsonlines(docs, SQUAD_ZEN + "preprocessd/documents.jsonl")
    write_jsonlines(dev_pairs, SQUAD_ZEN + "preprocessd/dev_paris.jsonl")
    write_jsonlines(train_pairs, SQUAD_ZEN + "preprocessd/train_paris.jsonl")
    

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

if __name__=="__main__":
    convert_to_jsonline()
    
    