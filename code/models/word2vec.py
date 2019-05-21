# load word2vec model
from gensim.models import Word2Vec

# build neural networks model using keras 
import keras 
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential,Model
from keras.layers import *
from keras import metrics
from keras.optimizers import * 
from keras.preprocessing.sequence import pad_sequences

# segmentation
import jieba_fast as jieba
jieba.load_userdict('model/data/Dictionary/product_AEs_names.txt')

import re
import numpy as np

# segment words iteratively
def segmentword(filepath):
  symbols ='[A-Za-z0-9，。；：‘’“”！？%$]*'
  with open(filepath, 'r') as data:
    for line in data.readlines():
      line = re.sub(symbols,'',line)
      line = re.sub(r'\n','',line)
      if line !='':
        words = list(jieba.cut(line, cut_all=False))
        yield np.array(words)

pos = segmentword('model/data/pos_neg_sample/pos_selected_meg.txt')
neg = segmentword('model/data/pos_neg_sample/neg_selected_meg.txt')

np.random.seed(100)
x_pos = np.array([words for words in pos])
y_pos = np.ones(len(x_pos))
pos_seq_max_len = max([len(words) for words in x_pos])
x_neg =np.array([words for words in neg])
y_neg = np.zeros(len(x_neg))
neg_seq_max_len = max([len(words) for words in x_neg])

x = np.concatenate([x_pos, x_neg])
y = np.append(y_pos, y_neg)

# permute dataset
num = len(y_pos) + len(y_neg)
idx = np.random.permutation(num)

x = x[idx]
y = y[idx]

max_seq_len = max([pos_seq_max_len, neg_seq_max_len])
embedding_size = 256

# train and test data split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, stratify = y)

# embed each word into a fixed length vector
# each sentence has been encoded into a matrix
w2v = Word2Vec.load('model/data/word2vec_model/wiki_zh.model')
words_weights = w2v.wv.syn0
vocab_size = words_weights.shape[0]
def word2idx(word):
  try:
    return w2v.wv.vocab[word].index
  except KeyError:
    return 0
  
def seq_iter(inputs):
  for sentence in inputs:
    if len(sentence)<max_seq_len:
      sentence = np.append(sentence, ['']*(max_seq_len-len(sentence)))
    words = np.array([word2idx(word) for word in sentence])
    yield words

X_train = np.array([words for words in seq_iter(x_train)])
X_test = np.array([words for words in seq_iter(x_test)])

#inputs = Input(shape = (max_seq_len, embedding_size))
model = Sequential()
model.add(Embedding(input_dim=vocab_size,
                   output_dim=embedding_size,
                   weights = [words_weights],
                   input_length = max_seq_len,
                   mask_zero =True,
                   trainable = False))
model.add(Bidirectional(LSTM(128)))
model.add(Dense(2, activation = 'softmax'))
model.compile(optimizer = 'adam',loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.summary()

history = model.fit(X_train, y_train, epochs = 5, batch_size = 100, 
                    validation_data = (X_test, y_test), verbose = 1)

y_test_predict = model.predict_classes(X_test)
from sklearn.metrics import confusion_matrix, classification_report

confusion_matrix(y_test, y_test_predict)
print(classification_report(y_test, y_test_predict))

# confusion matrix
# array([[2984,   33],
#       [  27,   44]])
#              precision    recall  f1-score   support
#         0.0       0.99      0.99      0.99      3017
#         1.0       0.57      0.62      0.59        71