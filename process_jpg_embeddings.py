import pandas as pd
import os
import numpy as np

import tensorflow as tf

filenames = os.listdir("./")
print(filenames)
print("num_files:", len(filenames))
raw_dataset = tf.data.TFRecordDataset(filenames)

patients = []
embeddings = []

c = 1
i = 0
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

    patients.append(int(filenames[i][0:8]))
    embeddings.append(embedding)

    c+=1
    i+=1

data_tuples = list(zip(patients,embeddings))
data_frame = pd.DataFrame(data_tuples, columns=['patient_id','cxr_jpg_embedding'])

print(data_frame)

data_frame.to_pickle("../cxr_jpg_embeddings.pkl")
