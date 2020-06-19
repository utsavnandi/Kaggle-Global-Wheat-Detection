
import re
import numpy as np
import pandas as pd

train_df = pd.read_csv('train.csv')

train_df['x'] = -1
train_df['y'] = -1
train_df['w'] = -1
train_df['h'] = -1

def expand_bbox(x):
    r = np.array(re.findall("([0-9]+[.]?[0-9]*)", x))
    if len(r) == 0:
        r = [-1, -1, -1, -1]
    return r

train_df[['x', 'y', 'w', 'h']] = np.stack(train_df['bbox'].apply(lambda x: expand_bbox(x)))
train_df.drop(columns=['bbox'], inplace=True)
train_df['x'] = train_df['x'].astype(np.float)
train_df['y'] = train_df['y'].astype(np.float)
train_df['w'] = train_df['w'].astype(np.float)
train_df['h'] = train_df['h'].astype(np.float)
train_df['area'] = (train_df['w'] * train_df['h'])/(1024*1024)
mapping = {'arvalis_1': 0, 'arvalis_2': 1, 'arvalis_3': 2, 'ethz_1': 3, 'inrae_1': 4, 'rres_1': 5, 'usask_1': 6}

def unique_count(df):
    for k in mapping:
        if len(df[df['source'] == k]['image_id'].unique()) > 0:
            print('unique images from '+k+' fold '+ str(mapping[k]) + ' ' + str(len(df[df['source'] == k]['image_id'].unique())))

print('combined:')
unique_count(train_df)
print()
train_df['folds'] = train_df.replace({'source': mapping}).source.values

valid_df = train_df[train_df['folds'].isin([1, 2, 4, 6])]   # <- Edit HERE for train split
train_df = train_df[train_df['folds'].isin([0, 3, 5])]      # <- Edit HERE for vaid split

print('train: ')
unique_count(train_df)
print('valid: ')
unique_count(valid_df)
print()
print('No. of train images', len(train_df['image_id'].unique()))
print('No. of test images', len(valid_df['image_id'].unique()))
print('split percent: ', len(valid_df['image_id'].unique())/(len(train_df['image_id'].unique())+len(valid_df['image_id'].unique())))

train_df.to_csv('./train_df.csv', index=False)
valid_df.to_csv('./valid_df.csv', index=False)
print('saved train_df and valid_df to current directory')