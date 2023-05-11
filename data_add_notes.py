import os
import tensorflow as tf
import numpy as np
import pandas as pd

path = 'processed'

# Opening Master Dataset

df_master = pd.read_csv(os.path.join(path, 'master_dataset_old.csv'))

# Processing notes

df_notes = pd.read_csv(os.path.join(path, 'aggregated_notes.csv'), index_col = 0)
df_notes.iloc[:,2] = df_notes['Notes'].str.replace('\n','')
df_notes.iloc[:,2] = df_notes['Notes'].str.replace('_','')
df_notes.iloc[:,2] = df_notes['Notes'].str.extract(r'IMPRESSION: (.*)')

df_notes = df_notes.dropna()

notes_id = df_notes['subject_id'].unique()

# Adding notes to the data sets

cxr_notes = pd.Series('', index = df_master.index)

unique_master = df_master.subject_id.value_counts()[df_master.subject_id.value_counts() == 1].index

unique_notes = df_notes.subject_id.value_counts()

total_added = 0

for index, row in df_master.iterrows():
    if(index % 10000 == 0):
        print(index)
    if(not row['subject_id'] in unique_master):
        continue
    if(not row['subject_id'] in notes_id):
        continue
    notes_add = df_notes[df_notes.subject_id == row['subject_id']]
    concat_notes = ' '.join(notes_add['Notes'])
    cxr_notes[index] = concat_notes
    total_added += 1

df_master['cxr_notes'] = cxr_notes
df_master['cxr'] = df_master['cxr_notes'] != ''
df_master['cxr'] = df_master['cxr'].astype(int)

# exporting the final dataset

df_master.to_csv(os.path.join(path, 'master_dataset.csv'), index = False)
    
