import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# read medrecon data

data = pd.read_csv(".\mimic-iv-ed-2.2\ed\medrecon.csv")

# helper function for splitting the description

def description(x):
    try:
        return x.split('-')[0].strip()
    except:
        print(x)
        return x
    
# only use the first category

data['etc_class'] = data['etcdescription'].apply(description)

data['etccode'] = data['etc_class']

# select top etc code

etccode_counts = data['etccode'].value_counts()
top_etc = list(etccode_counts.index[0:31])
is_top_etc = data['etccode'].isin(top_etc)

data['etccode'][is_top_etc == False] = -1

data['etccode_str'] = data['etccode'].astype(str)

stay_med = data.groupby('stay_id')['etccode_str'].apply(','.join).reset_index()

top_etc = [str(i) for i in top_etc]
top_etc.append('Others')

stay_med_processed = pd.DataFrame(index = stay_med['stay_id'], columns = top_etc).fillna(0)

# one-hot encode the drug code

for i in tqdm(range(0, len(data))):
    etccode = data['etccode'][i]
    stay_id = data['stay_id'][i]
    stay_med_processed.loc[stay_id][etccode] = 1

stay_med_processed.to_csv('stay_with_med_code.csv')
stay_med_processed.columns = ["med_"+i for i in stay_med_processed.columns]

# add the embedding to training and testing

train = pd.read_csv('processed_train_test/train_old.csv', index_col=0)
train_med = train.set_index('stay_id').join(stay_med_processed)
train_med.iloc[:,127:] = train_med.iloc[:,127:].fillna(0)
train_med.to_csv('processed_train_test/train.csv')

test = pd.read_csv('processed_train_test/test_old.csv', index_col=0)
test_med = test.set_index('stay_id').join(stay_med_processed)
test_med.iloc[:,127:] = test_med.iloc[:,127:].fillna(0)
test_med.to_csv('processed_train_test/test.csv')
