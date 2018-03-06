##LSTM

import gensim

##
import os

os.chdir('/Users/whatsoever/Dropbox/101 share')

## import validation and train

import pickle

with open('validation_set.p', 'r') as f:
    validation_set = pickle.load(f)
    
    
with open('training_set.p', 'r') as f:
    training_set = pickle.load(f)

## import w2v model
#model = gensim.models.Word2Vec.load_word2vec_format('/Users/mac/Downloads/knowledge-vectors-skipgram1000.bin', binary=True)

#model = gensim.models.Word2Vec.load_word2vec_format('/Users/mac/Downloads/GoogleNews-vectors-negative300.bin', binary=True)
model = gensim.models.Word2Vec.load('w2v_model_context_training_wiki')

##getVec: string -> vector (to get Y)
import numpy as np
import re

import nltk
stopwords = set(nltk.corpus.stopwords.words())

stemmer = nltk.PorterStemmer()

def stem(l):
  return [stemmer.stem(x.lower()) for x in l]

def mean(s):
    l = s.split()
    l = stem(l)
    vecs = []
    for x in l:
        if len(x) == 0 or x not in model or x in stopwords:
            continue
        vecs.append(model[x])
    if len(vecs) == 0:
        return None
    else:
        return np.array(vecs, dtype=np.float64).mean(axis=0)
        
def getVec(s, use_keyword = True, default_vec = np.array([0.5 for i in range(300)])):
    s.replace('-', ' ')
    s = re.sub(r'[^\x00-\x7F]+',' ', s)
    if not use_keyword:
        vec = mean(s)
        return vec if vec != None else default_vec
    keywords = rake.extract(s, incl_scores=True)
    keywords = [(mean(w[0]), w[1]) for w in keywords]
    keywords = [w for w in keywords if w[0] != None]
    total_score = sum([w[1] for w in keywords])
    #keywords = [w[0] * (w[1] / total_score) for w in keywords]
    #return sum(keywords) if len(keywords) != 0 else default_vec
    keywords = [w[0] for w in keywords]
    return np.array(keywords).mean(axis=0) if len(keywords) != 0 else default_vec

## get training data
windows_length = 152;

def getVecList(s, default_vec = np.array([0.5 for i in range(300)])):
    s.replace('-', ' ')
    s = re.sub(r'[^\x00-\x7F]+',' ', s)
    l = s.split()
    l = stem(l)
    vecs = []
    for x in l:
        if len(x) == 0 or x not in model or x in stopwords:
            continue
        vecs.append(model[x])
    if len(vecs) == 0:
        vecs.append(default_vec)
    return vecs
    

def getWindows(l, window_size = 5, empty_vec = np.array([0 for i in range(300)])):
    r = []
    for i in range(len(l)):
        r.append(getWindow(l, i, window_size, empty_vec))
    return repeatWindow(r)
    
def getWindow(l, i, window_size, empty_vec):
    start_index = i - window_size/2
    end_index = i + window_size/2
    w = []
    for j in range(start_index, end_index+1):
        if j < 0 or j >= len(l):
            w.append(empty_vec)
        else:
            w.append(l[j])
    return np.array(w, dtype=np.float64).flatten()

def repeatWindow(l):
    i = 0;
    r = [];
    while len(r) != windows_length:
        r.append(l[i])
        i = (i+1) % len(l)
    return r

##

train = training_set[:-500]
test = training_set[-500:]

X_train = np.array([getWindows(getVecList(x['question'])) for x in train], dtype=np.float64)
Y_train = np.array([getVec(x['answers'][x['answer_id']], use_keyword = False) for x in train], dtype=np.float64)


##
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedMerge
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.datasets import imdb

##

# Embedding
max_features = 20000
maxlen = 300
embedding_size = 128

# Convolution
filter_length = 3
nb_filter = 64
pool_length = 2

# LSTM
lstm_output_size = 300

# Training
batch_size = 30
nb_epoch = 2

##model 
lstm_model = Sequential()
#model.add(Embedding(45900, 300)) 
#model.add(Dropout(0.25))
#model.add(Convolution1D(nb_filter=nb_filter, filter_length=filter_length, border_mode='valid', activation='relu', subsample_length=1))
#model.add(MaxPooling1D(pool_length=pool_length))
lstm_model.add(LSTM(1500, input_dim=1500, return_sequences=False))
lstm_model.add(Dense(300))
lstm_model.add(Activation('softmax'))
#model.add(Dense(1))
#model.add(Activation('sigmoid'))

#model.compile(loss='binary_crossentropy', optimizer='adam',class_mode='binary')

lstm_model.compile(loss='mean_squared_error', optimizer='sgd')

##

print('Train...')
lstm_model.fit(X_train, Y_train, nb_epoch=2, batch_size=30)

##

json_string = lstm_model.to_json()
open('my_model_architecture.json', 'w').write(json_string)
lstm_model.save_weights('my_model_weights.h5')

##
# elsewhere...
model = model_from_json(open('my_model_architecture.json').read())
model.load_weights('my_model_weights.h5')

##
#X_test = np.array([getWindows(getVecList(x['question'])) for x in validation_set], dtype=np.float64)

##
#Y_test = lstm_model.predict(np.array([getWindows(getVecList(x['question'])) for x in validation_set[1:2]], dtype=np.float64))

##
def lstm_predict(question, answers):
    question_v = lstm_model.predict(np.array([getWindows(getVecList(x)) for x in [question]], dtype=np.float64))[0]
    answers_v = [getVec(x, use_keyword = False) for x in answers]
    answers_s = [np.dot(y, question_v) for y in answers_v]
    max_index, max_value = max(enumerate(answers_s), key=operator.itemgetter(1))
    return max_index

import operator
import sys

def get_correctness(r = 0.8, default_s=0.5):
    corrent_count = 0;
    i = 0
    for x in test:
        print(i)
        i += 1
        max_index = lstm_predict(x['question'], x['answers'])
        if (max_index == x['answer_id']):
            corrent_count += 1
            print('T')
        else:
            print('F')
    return float(corrent_count)/len(test)
    
print(get_correctness())
    
##
import operator
import sys

m = ['A', 'B', 'C', 'D']

output = []

i = 0
for x in validation_set:
    print(i)
    i += 1
    question_v = lstm_model.predict(np.array([getWindows(getVecList(s)) for s in [x['question']]], dtype=np.float64))[0]
    answers_v = [getVec(y, use_keyword = False) for y in x['answers']]
    answers_s = [np.dot(y, question_v) for y in answers_v]
    max_index, max_value = max(enumerate(answers_s), key=operator.itemgetter(1))
    output.append({'id': x['id'], 'correctAnswer': m[max_index]})