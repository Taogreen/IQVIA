from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import xgboost as xgb
d2v = Doc2Vec.load('model/data/Doc2Vec_oncology/doc2vec.model')

import jieba_fast as jieba
jieba.load_userdict('model/data/product_AEs_names.txt')
import numpy as np
import re
puncts = "[\s+\.\!\/_,$\-;\:%^*(+\"\']+|[+——！，。？、；‘’“”《》【】~@#￥%……&*（）]+"

def segment(filepath):
  with open(filepath,'r') as f:
    for line in f.readlines():
      sen = re.sub(puncts,"",line)
      sen = re.sub(r'\n','',sen)
      if sen !='':
        yield list(jieba.cut(sen, cut_all = False))
      

###########################################################
######################### TFIDF ###########################
###########################################################
iter_pos = segment('model/data/pos_neg_sample/positive.txt')
iter_neg = segment('model/data/pos_neg_sample/negative.txt')

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
vectorizer = CountVectorizer()
transformer = TfidfTransformer()
words_pos = np.array([" ".join(line) for line in iter_pos])
words_neg = np.array([" ".join(line) for line in iter_neg])
words = np.concatenate([words_pos, words_neg])
y_pos = np.ones(len(words_pos), dtype = int)
y_neg = np.zeros(len(words_neg), dtype = int)
y = np.concatenate([y_pos, y_neg])
np.random.seed(10)
index = np.arange(len(y))
np.random.shuffle(index)
words,y = words[index],y[index]

#k = sorted(cos_dis[7], reverse = True)
#for i, val in enumerate(cos_dis[7]):
#  if val >= min(k[:6]):
#    print (words[i])
from sklearn.model_selection import train_test_split
words_train, words_test, y_train, y_test = train_test_split(words, y, test_size=0.2, random_state=30, stratify=y)

vec = vectorizer.fit_transform(words_train)
trans = transformer.fit_transform(vec)
x_train = trans.toarray()

vec1 = vectorizer.transform(words_test)
trans1 = transformer.transform(vec1)
x_test = trans1.toarray()

scale_pos_weight = sum(1-y_train)/sum(y_train)
params = {
    'booster':'gbtree',
    'objective': 'binary:logistic',
    'silent': 1,
    'eval_metric': 'aucpr',
    'n_estimators': 100,
    'n_jobs' : 8,
    'scale_pos_weight': scale_pos_weight
  }
bst = xgb.XGBClassifier(**params)
bst.fit(x_train, y_train, eval_set = [(x_train, y_train), (x_test, y_test)])
y_test_predict = bst.predict(x_test)
from sklearn.metrics import confusion_matrix, classification_report
confusion_matrix(y_test, y_test_predict)
# array([[2848,  169],
#       [  16,   55]])
print(classification_report(y_test, y_test_predict))
#              precision    recall  f1-score   support
#
#           0       0.99      0.94      0.97      3017
#           1       0.25      0.77      0.37        71


from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 2)
x_train_over, y_train_over = sm.fit_resample(x_train, y_train)

from imblearn.under_sampling import RepeatedEditedNearestNeighbours
rus = RepeatedEditedNearestNeighbours(random_state=42)
x_train_under, y_train_under = rus.fit_resample(x_train, y_train)

scale_pos_weight = sum(1-y_train_over)/sum(y_train_over)
params = {
    'booster':'gbtree',
    'objective': 'binary:logistic',
    'silent': 1,
    'eval_metric': 'aucpr',
    'n_estimators': 100,
    'n_jobs' : 8,
    'scale_pos_weight': scale_pos_weight
  }
bst_over = xgb.XGBClassifier(**params)
bst_over.fit(x_train_over, y_train_over, eval_set = [(x_train, y_train), (x_test, y_test)])
y_test_pred_tfidf = bst_over.predict(x_test)
proba = bst_over.predict_proba(x_test)
np.save('model/data/TFIDF-XGB-pred.npy', proba)
np.save('model/data/TFIDF-XGB-original.npy', y_test)
from sklearn.metrics import confusion_matrix, classification_report
confusion_matrix(y_test,y_test_pred_tfidf)
#array([[2894,  123],
#       [  26,   45]])
print(classification_report(y_test,y_test_pred_tfidf))
#              precision    recall  f1-score   support
#           0       0.99      0.96      0.97      3017
#           1       0.27      0.63      0.38        71




from sklearn.metrics.pairwise import cosine_similarity as cos
from sklearn.metrics import f1_score
cos_m = cos(x_test, x_train)
idx_max = cos_m.argmax(1)
y_pred = y_train[idx_max]
print('F1 score is ', f1_score(y_test, y_pred, average = 'binary'))
confusion_matrix(y_test, y_pred)



scale_pos_weight = sum(1-y_train_under)/sum(y_train_under)
params = {
    'booster':'gbtree',
    'objective': 'binary:logistic',
    'silent': 1,
    'eval_metric': 'aucpr',
    'n_estimators': 100,
    'n_jobs' : 8,
    'scale_pos_weight': scale_pos_weight
  }
bst_under = xgb.XGBClassifier(**params)
bst_under.fit(x_train_under, y_train_under, eval_set = [(x_train, y_train), (x_test, y_test)])






###########################################################
######################### Doc2vec #########################
###########################################################
iter_pos = segment('model/data/pos_neg_sample/positive.txt')
iter_neg = segment('model/data/pos_neg_sample/negative.txt')
x_pos = np.array([d2v.infer_vector(it) for it in iter_pos])
y_pos = np.ones(len(x_pos), dtype = int)
x_neg = np.array([d2v.infer_vector(it) for it in iter_neg])
y_neg = np.zeros(len(x_neg), dtype = int)

np.random.seed(10)
x = np.vstack([x_pos, x_neg])
y = np.concatenate([y_pos, y_neg])
index = np.arange(len(y))
np.random.shuffle(index)
x,y = x[index],y[index]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 30, stratify = y)

from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=2)
x_train_over, y_train_over= sm.fit_resample(x_train, y_train)

from imblearn.under_sampling import RepeatedEditedNearestNeighbours
rus = RepeatedEditedNearestNeighbours(random_state=42)
x_train_under, y_train_under = rus.fit_resample(x_train, y_train)


import xgboost as xgb
scale_pos_weight = sum(1-y_train_over)/sum(y_train_over)
params = {
    'booster':'gbtree',
    'objective': 'binary:logistic',
    'silent': 1,
    'eval_metric': 'aucpr',
    'n_estimators': 100,
    'n_jobs' : 8,
    'scale_pos_weight': scale_pos_weight
  }
clf_over = xgb.XGBClassifier(**params)
clf_over.fit(x_train_over, y_train_over, eval_set=[(x_train, y_train), (x_test, y_test)])
y_test_pred = clf_over.predict(x_test)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_test_pred)
#array([[2999,   18],
#       [  40,   31]])
from sklearn.metrics import classification_report as cr
print(cr(y_test, y_test_pred))
#              precision    recall  f1-score   support
#           0       0.99      0.99      0.99      3017
#           1       0.63      0.44      0.52        71

scale_pos_weight = sum(1-y_train_under)/sum(y_train_under)
params = {
    'booster':'gbtree',
    'objective': 'binary:logistic',
    'silent': 1,
    'eval_metric': 'aucpr',
    'n_estimators': 100,
    'n_jobs' : 8,
    'scale_pos_weight': scale_pos_weight
  }
clf_under = xgb.XGBClassifier(**params)
clf_under.fit(x_train_under, y_train_under, eval_set = [(x_train, y_train), (x_test, y_test)])
y_test_pred_1 = clf_over.predict(x_test)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_test_pred_1)
#array([[2810,    9],
#       [  19,    3]])
from sklearn.metrics import classification_report as cr
print(cr(y_test, y_test_pred_1))
#              precision    recall  f1-score   support
#           0       0.99      1.00      1.00      2819
#           1       0.25      0.14      0.18        22



scale_pos_weight = sum(1-y_train)/sum(y_train)
params = {
    'booster':'gbtree',
    'objective': 'binary:logistic',
    'silent': 1,
    'eval_metric': 'aucpr',
    'n_estimators': 100,
    'n_jobs' : 8,
    'scale_pos_weight': scale_pos_weight
  }
clf = xgb.XGBClassifier(**params)
clf.fit(x_train, y_train, eval_set = [(x_train, y_train), (x_test, y_test)])
y_test_pred_2 = clf.predict(x_test)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_test_pred_2)
#array([[3013,    4],
#       [  50,   21]])
from sklearn.metrics import classification_report as cr
print(cr(y_test, y_test_pred_2))
#              precision    recall  f1-score   support
#           0       0.98      1.00      0.99      3017
#           1       0.84      0.30      0.44        71





###########################################################
########################## BERT ###########################
###########################################################
import numpy as np
import xgboost as xgb
x_pos = np.load('model/data/BERT_output/leukemia/bert_pos_0422.npy')
y_pos = np.ones(len(x_pos), dtype = int)

x_neg = np.load('model/data/BERT_output/leukemia/bert_neg_0422.npy')
y_neg = np.zeros(len(x_neg), dtype = int)

x = np.concatenate([x_pos, x_neg])
y = np.concatenate([y_pos, y_neg])
np.random.seed(100)
index = np.arange(len(y))
np.random.shuffle(index)
x,y = x[index], y[index]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.1, random_state = 100, stratify = y)

scale_pos_weight = sum(1-y_train)/sum(y_train)
params = {
    'booster':'gbtree',
    'objective': 'binary:logistic',
    'silent': 1,
    'eval_metric': 'aucpr',
    'n_estimators': 100,
    'n_jobs' : 8,
    'scale_pos_weight': scale_pos_weight
  }
clf = xgb.XGBClassifier(**params)
clf.fit(x_train, y_train, eval_set = [(x_train, y_train), (x_test, y_test)])


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report as cr
y_test_predict = clf.predict(x_test)
confusion_matrix(y_test, y_test_predict)
print(cr(y_test, y_test_predict))

y_predict = clf.predict(x)
confusion_matrix(y, y_predict)
print(cr(y, y_predict))
ind = [i for i in range(len(x)) if y[i]==1 and y_predict[i]==0]
sorted(index[ind])

#array([[2928,   89],
#       [  18,   53]])

#              precision    recall  f1-score   support
#           0       0.99      0.97      0.98      3017
#           1       0.37      0.75      0.50        71


from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 2)
x_train_over, y_train_over = sm.fit_resample(x_train, y_train)

scale_pos_weight = sum(1-y_train_over)/sum(y_train_over)
params = {
    'booster':'gbtree',
    'objective': 'binary:logistic',
    'silent': 1,
    'eval_metric': 'aucpr',
    'n_estimators': 100,
    'n_jobs' : 8,
    'learning_rate':.2,
    'scale_pos_weight': scale_pos_weight
  }
clf_over = xgb.XGBClassifier(**params)
clf_over.fit(x_train_over, y_train_over, eval_set = [(x_train, y_train), (x_test, y_test)])


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report as cr
y_test_predict = clf_over.predict(x_test)
proba1 = clf_over.predict_proba(x_test)
np.save('model/data/BERT-XGB-pred', proba1)
np.save('model/data/BERT-XGB-original', y_test)
confusion_matrix(y_test, y_test_predict)
print(cr(y_test, y_test_predict))
#array([[2879,  138],
#       [  18,   53]])
#              precision    recall  f1-score   support
#           0       0.99      0.95      0.97      3017
#           1       0.28      0.75      0.40        71
print('sample size: ', len(x))
print('positive ratio overall:%.5f'% (sum(y)/len(y)))
print('Training sample size:%d, its positive ratio:%.5f'%(len(x_train), sum(y_train)/len(y_train)))
print('Testing sample size:%d, its positive ratio:%.5f'%(len(x_test), sum(y_test)/len(y_test)))