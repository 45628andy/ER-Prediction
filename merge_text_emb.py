import os
import time
import random
import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import optimizers, metrics
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from helpers import PlotROCCurve

import torch
from transformers import AutoModel, AutoTokenizer

from tqdm import tqdm

# Load the model and tokenizer
url = "microsoft/BiomedVLP-CXR-BERT-specialized"
tokenizer = AutoTokenizer.from_pretrained(url, trust_remote_code=True)
model = AutoModel.from_pretrained(url, trust_remote_code=True)

max_len = 2400

train = pd.read_csv(('train.csv'))
embs = []
for _, row in tqdm(train.iterrows()):
    if row.cxr == 0:
        # print('non cxr')
        embs.append(np.zeros(128))
    else:
        # print('cxr')
        notes = row.cxr_notes
        notes = notes.strip()
        sep_notes = notes.split('.')
        com_notes = ''
        for note in sep_notes:
            note = note.strip()
            note += '. '
            if len(com_notes) + len(note) <= max_len:
                com_notes += note
            else:
                break
        tokenizer_output = tokenizer.batch_encode_plus(batch_text_or_text_pairs=[com_notes],
                                                    add_special_tokens=True,
                                                    padding='longest',
                                                    return_tensors='pt')
        embeddings = model.get_projected_text_embeddings(input_ids=tokenizer_output.input_ids,
                                                        attention_mask=tokenizer_output.attention_mask)   
        embs.append(embeddings[0].detach().numpy())

embs = np.array(embs)
np.save("train_cxr_note_embs.npy", embs)


test = pd.read_csv(('test.csv'))
embs = []
for _, row in tqdm(test.iterrows()):
    if row.cxr == 0:
        # print('non cxr')
        embs.append(np.zeros(128))
    else:
        # print('cxr')
        notes = row.cxr_notes
        notes = notes.strip()
        sep_notes = notes.split('.')
        com_notes = ''
        for note in sep_notes:
            note = note.strip()
            note += '. '
            if len(com_notes) + len(note) <= max_len:
                com_notes += note
            else:
                break
        tokenizer_output = tokenizer.batch_encode_plus(batch_text_or_text_pairs=[com_notes],
                                                    add_special_tokens=True,
                                                    padding='longest',
                                                    return_tensors='pt')
        embeddings = model.get_projected_text_embeddings(input_ids=tokenizer_output.input_ids,
                                                        attention_mask=tokenizer_output.attention_mask)   
        embs.append(embeddings[0].detach().numpy())

embs = np.array(embs)
np.save("test_cxr_note_embs.npy", embs)