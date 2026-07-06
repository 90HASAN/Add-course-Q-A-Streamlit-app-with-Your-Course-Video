import os
import json
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

torch.set_num_threads(4)

model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")


def createEmbeddings(text_list):
    embeddings = model.encode(
        text_list,
        batch_size=64,
        show_progress_bar=True
    )
    return embeddings.tolist()  # convert numpy arrays -> plain lists for JSON/DataFrame storage


jsons = os.listdir("json")
my_dict = []
chunk_id = 0

for json_file in jsons:
    with open(f"json/{json_file}") as f:
        content = json.load(f)
    embeddings = createEmbeddings([c['text'] for c in content['chunks']])

    for i, chunk in enumerate(content['chunks']):
        chunk['chunk_id'] = chunk_id
        chunk['embedding'] = embeddings[i]

        my_dict.append(chunk)
        chunk_id += 1
"""         if i == 3:
            break """

df = pd.DataFrame.from_records(my_dict)

joblib.dump(df, 'embaddings.joblib')