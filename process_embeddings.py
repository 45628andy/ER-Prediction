import pandas as pd
import os
import numpy as np

df_train = pd.read_csv((os.path.join('../', 'train.csv')))
df_test = pd.read_csv((os.path.join('../', 'test.csv')))
patient_ids = pd.concat([df_train.loc[df_train.cxr==1]['subject_id'], df_test.loc[df_train.cxr==1]['subject_id']])
patient_ids = np.sort(patient_ids)
patient_ids = patient_ids.tolist()

#print(patient_ids)
#print(type(patient_ids))

import tensorflow as tf
import os

filenames = os.listdir("./")
print("num_files:", len(filenames))
raw_dataset = tf.data.TFRecordDataset(filenames)
raw_dataset

patients = []
embeddings = []

c = 1
for raw_record in raw_dataset.take(len(filenames)):
    print("file:", c)
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())

    result = {}
    for key, feature in example.features.feature.items():
        kind = feature.WhichOneof('kind')
        result[key] = np.array(getattr(feature, kind).value)

    embedding = result['embedding']
    #print(embedding)

    patient_id = int((result['image/id'][0][42:50]).decode())
    #print(patient_id)

    if patient_id in patient_ids and patient_id not in patients:
        #print("in")
        patients.append(patient_id)
        embeddings.append(embedding)

    c+=1

data_tuples = list(zip(patients,embeddings))
data_frame = pd.DataFrame(data_tuples, columns=['patient_id','cxr_embedding'])

data_frame.to_pickle("../dummy.pkl")
