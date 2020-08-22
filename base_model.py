#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 12:15:41 2020

@author: jialiluan
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
from collections import Counter, defaultdict #defaultdict provides value of nonexist key

BOOK_DIR = "/Users/jialiluan/Downloads/goodreads_books_comics_graphic.json"

data = []
for line in open(BOOK_DIR, 'r'):
    data.append(json.loads(line))

keys_to_remove = ['popular_shelves', 
                  'description',
                  'kindle_asin',
                  'similar_books',
                  'link',
                  'url',
                  'image_url',
                  'work_id',
                  'title_without_series',
                  'asin',
                  'isbn13',
                  'isbn',
                  'edition_information',
                  'series'
                  ]
for item in data:  # my_list if the list that you have in your question
    for key in keys_to_remove:
        del item[key]
        
data_df = pd.DataFrame(data)
data_df['authors'] = data_df['authors'].str[0]

for item in data_df['authors']:
    del item['role']
   

#data_df['is_ebook'] = data_df['is_ebook'].map({'true': 1, 'false':0})

data_df['authors'] = [i for v in data_df['authors'].values for i in v.values()]

#data_df['series'] = data_df['series'].apply(lambda y: np.nan if len(y)==0 else y)
#data_df = data_df.replace(r'\s+',np.nan,regex=True).replace('',np.nan)

# l = []
# for v in data_df['authors'].values:
#     for i in v.values():
#         l.append(i)
# data_df['authors'] = l

data_df = data_df.apply(pd.to_numeric,errors='ignore')
book_title = data_df['title']
data_df = data_df.drop(['title'], axis = 1)
# create a regressor object 
from sklearn.preprocessing import LabelEncoder
# creating instance of labelencoder
le= LabelEncoder()
# Assigning numerical values and storing in another column

def fit_transform_ohe(df,col_name):
    """This function performs one hot encoding for the specified
column.
    Args:
        df(pandas.DataFrame): the data frame containing the mentioned column name
        col_name: the column to be one hot encoded
    Returns:
        tuple: label_encoder, one_hot_encoder, transformed column as pandas Series
    """
    # label encode the column
    le = preprocessing.LabelEncoder()
    le_labels = le.fit_transform(df[col_name])
    df[col_name] = le_labels
    
    
data_df_object = data_df.select_dtypes('object').columns.tolist()

for i in data_df_object:
    fit_transform_ohe(data_df,i)

for column in data_df.columns:
    data_df[column].fillna(data_df[column].mean(), inplace=True)

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
X_all = data_df.drop(['average_rating','country_code'], axis=1)
Y_all = data_df['average_rating']


X_train, X_test, Y_train, Y_test = train_test_split(X_all, Y_all,test_size=0.2,random_state=42)


dtm = DecisionTreeRegressor(max_depth=4,
                           min_samples_split=5,
                           max_leaf_nodes=10)


dtm_fit = dtm.fit(X_train,Y_train)

from sklearn.model_selection import cross_val_score
dtm_scores = cross_val_score(dtm_fit, X_train, Y_train, cv = 5)

print("mean cross validation score: {}".format(np.mean(dtm_scores)))
print("score without cv: {}".format(dtm_fit.score(X_train, Y_train)))

from sklearn.metrics import r2_score
print(dtm_fit.score(X_test, Y_test))

y_pred = np.around(dtm.predict(X_test),2)

Y_test = Y_test.to_numpy()
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import accuracy_score
mean_squared_error(Y_test, y_pred)


y
y_pred_train = dtm.predict(X_train)
mean_squared_error(Y_train, y_pred_train)



