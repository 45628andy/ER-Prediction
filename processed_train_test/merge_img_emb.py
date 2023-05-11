import pickle
import numpy as np
import pandas as pd

file = 'cxr_jpg_emb.pkl'

with open(str(file), 'rb') as f:
    embeddings = pickle.load(f)


emb_dict = {}


for id_, emb in zip(embeddings['patient_id'], embeddings['cxr_jpg_embedding']):
    emb_dict[str(id_)] = emb


train = pd.read_csv('train.csv')
img_emb = []
num_in = 0
for index, row in train.iterrows():
    if row['cxr'] and str(row['subject_id']) in emb_dict.keys():
        img_emb.append(emb_dict[str(row['subject_id'])])
        num_in += 1
    else:
        img_emb.append(np.zeros(1376))
img_emb = np.array(img_emb)
np.save("train_cxr_img_embs.npy", img_emb)


test = pd.read_csv('test.csv')
img_emb = []
num_in = 0
for index, row in test.iterrows():
    if row['cxr'] and str(row['subject_id']) in emb_dict.keys():
        img_emb.append(emb_dict[str(row['subject_id'])])
        num_in += 1
    else:
        img_emb.append(np.zeros(1376))
img_emb = np.array(img_emb)
np.save("test_cxr_img_embs.npy", img_emb)