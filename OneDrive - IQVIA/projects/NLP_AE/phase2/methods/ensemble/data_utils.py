import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def text2str(path):
    sentences = []
    with open(path, 'r', encoding = 'utf-8') as f:
        for line in f.readlines():
          line = line.strip()
          if line !='':
            sentences.append(line)
        f.close()
    return sentences
  
def train_test_split(x, y, rs = 100, test_size = 0.1):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size,\
                                                        random_state = rs, stratify=y)
    return x_train, x_test, y_train, y_test

def overSampling(x_train, y_train):
    sm = SMOTE(random_state=2)
    x_train, y_train  = sm.fit_sample(x_train, y_train)
    
    return x_train, y_train