import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def label_encode(ds):
    for col in ds.columns.values:
        le=LabelEncoder()
        unique_val=list(ds[col].unique())
        le.fit(unique_val)
        print(col, le.classes_)
        col_val = list(ds[col].values)
        new_col_val = le.transform(col_val)
        ds[col] = new_col_val
    #display(ds.head())
        

#ratio is percentage of data that is training data
def get_train_test(ds, ratio):
    test, train = train_test_split(ds, test_size=ratio, random_state=42)
    #print(test.shape)
    #print(train.shape)
    return test, train
  
dataset = pd.read_csv('./dataset_encoded.csv')
#print(dataset.mode()['stalk-root'][0])
#print(dataset.shape)
for index, row in dataset.iterrows():
    if(row['stalk-root'] == 0): row['stalk-root']=1
#print(dataset.shape)
#label_encode(dataset)
test3, train7 = get_train_test(dataset, 0.7)
test5, train5 = get_train_test(dataset, 0.5)
test3.to_csv('test0.3_replace.csv', index=False)
train7.to_csv('train0.7_replace.csv', index=False)
test5.to_csv('test0.5_replace.csv', index=False)
train5.to_csv('train0.5_replace.csv', index=False)

