#!/usr/bin/env python
# coding: utf-8

# In[1]:


from data_loader import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from keras.models import Model
from keras.layers import Dense, Embedding, Input, LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import roc_auc_score


# In[2]:


train = load_train_data("data/train.csv")


# In[3]:


test_ids, test_texts, test_labels = load_test_data("data/test.csv", "data/test_labels.csv", False)


# In[4]:


train_labels = train[0].iloc[:,2:]


# In[5]:


dim = 150
vol_size = 20000
emb_size = 128


# In[6]:


X = train[0].comment_text
X_val = train[1].comment_text
Y = train[0][["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
Y_val = train[1][["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values


# In[7]:


tok = Tokenizer(num_words = vol_size)
tok.fit_on_texts(list(X))
tok.fit_on_texts(list(X_val))


# In[8]:


X = tok.texts_to_sequences(X)
X_val = tok.texts_to_sequences(X_val)
X_test = tok.texts_to_sequences(test_texts)
X = sequence.pad_sequences(X, maxlen = dim)
X_val = sequence.pad_sequences(X_val, maxlen = dim)
X_test = sequence.pad_sequences(X_test, maxlen = dim)


# In[ ]:


def lstm_model(layer_num = 1):
    _in = Input(shape = (dim, ))
    layer = Embedding(vol_size, emb_size)(_in)
    for i in range(layer_num):
        layer = Bidirectional(LSTM(50, return_sequences = True, recurrent_dropout = 0.15))(layer)
    layer = GlobalMaxPool1D()(layer)
    layer = Dropout(0.2)(layer)
    layer = Dense(50, activation = 'relu')(layer)
    layer = Dropout(0.2)(layer)
    layer = Dense(6, activation = 'sigmoid')(layer)
    model = Model(inputs = _in, outputs = layer)
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])
    return model


# In[ ]:


model = lstm_model(layer_num = 2)


# In[ ]:


stop = EarlyStopping(monitor = 'val_loss', patience = 1)


# In[ ]:


res = model.fit(X, Y, batch_size = 512, epochs = 2, validation_data = (X_val, Y_val), callbacks = [stop])


# In[ ]:


Y_test = model.predict(X_test)


# In[ ]:


sum = 0
for i in range(6):
    score = roc_auc_score(test_labels[:,i], Y_test[:,i])
    sum += score


# In[ ]:


print (sum / 6)


# In[ ]:




