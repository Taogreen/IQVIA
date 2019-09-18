__author__ = 'Wentao Deng'
import sys, os
sys.path.append(os.path.abspath('.'))

from ensemble.data_utils import text2str, train_test_split, overSampling
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model, model_from_json
from keras.utils import to_categorical
from keras.layers import Dense, multiply, Input, Conv1D, Activation, MaxPool1D, Concatenate, Dropout, Bidirectional, LSTM

import numpy as np
np.random.seed(100) # for replication
import logging

from ensemble.bert_start import bc
class AETrain(object):
  
    logger = logging.getLogger(__name__)
    
    def __init__(self, pos_path, neg_path, **kwarg):
        super(AETrain, self).__init__()
        assert pos_path.endswith('.txt'), 'txt file required.'
        assert neg_path.endswith('.txt'), 'txt file required.'
        
        self.pos_sen = text2str(pos_path)
        self.neg_sen = text2str(neg_path)

    
    def BERTEncoding(self):
        '''
        Start BERT service to encode sentences into fixed length vectors.
        Labels will be prepared at the same time
        '''
        self.logger.info('Start encoding...')
        X_pos = [c for c in bc(self.pos_sen)]
        X_pos = [c for c in bc(self.pos_sen)]
        X_pos, X_neg = np.asarray(X_pos), np.asarray(X_neg)
        self.logger.debug(f'Embedding length of X_pos or X_neg is {X_pos.shape[1]}(1536)')
        
        ## Prepare Labels ## 
        y_pos = np.ones(len(X_pos), dtype = int)
        y_neg = np.zeros(len(X_neg), dtype = int)
        X = np.concatenate([X_pos, X_neg], axis = 0)
        y = np.concatenate([y_pos, y_neg], axis = 0)

        # shuffle data
        index = np.arange(len(y))
        np.random.shuffle(index)
        X,y = X[index], y[index]
        
        return X,y
    
    def modelTrain(self,X, y, test = False):
        x_train, y_train = X,y
        if test:
            self.logger.info('Start train/test split...')
            x_train, x_test, y_train, y_test = train_test_split(X,y)
            X_test = np.reshape(x_test, (len(x_test),24, 64))
            Y_test = to_categorical(y_test)
            
        self.logger.info('Start over sampling...')
        x_train, y_train = overSampling(x_train, y_train)
        X_train = np.reshape(x_train,(len(x_train), 24, 64))
        Y_train = to_categorical(y_train)
       

        inputs = Input(shape = (24,64), name = 'input')
        conv1_1 = Conv1D(128, kernel_size=2, padding = 'valid', strides = 1)(inputs)
        activation1 = Activation('relu')(conv1_1)
        maxpool1 = MaxPool1D(3)(activation1)

        conv2_1 = Conv1D(128, kernel_size=3, padding = 'valid', strides = 1)(inputs)
        activation2 = Activation('relu')(conv2_1)
        maxpool2 = MaxPool1D(3)(activation2)

        conv3_1 = Conv1D(128, kernel_size=4, padding = 'valid', strides = 1)(inputs)
        activation3 = Activation('relu')(conv3_1)
        maxpool3 = MaxPool1D(3)(activation3)

        concat = Concatenate(axis = 1)([maxpool1, maxpool2, maxpool3])
        dropout = Dropout(0.6)(concat)
        lstm_1 = Bidirectional(LSTM(100), name = 'bilstm')(dropout)
        fc1 = Dense(200, activation='softmax', name = 'fc1')(lstm_1)
        attention = multiply([lstm_1,fc1], name='attention_mul')
 
        output = Dense(2, activation = 'softmax', name='output')(attention)
        m = Model(input=inputs, output = output)
        m.summary()
        m.compile(optimizer='adam', 
                      loss='categorical_crossentropy', metrics=['accuracy'])
        m.fit(X_train, Y_train, batch_size = 100, epochs = 20, verbose = 2)
        if test:
            m.fit(X_train, Y_train, batch_size = 100, epochs = 20,\
                  validation_data = (X_test, Y_test))
        self.model = m
        return m
      
    if __name__ == '__main__':
        import time
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(name)-12s %(levelname)-8s %(funcName)s() %(message)s')
        basicDir = './training_data/oncology/'
        path_pos = basicDir + 'pos_0815.txt'
        path_neg = basicDir + 'neg_0815.txt'
        
        train = AETrain(path_pos, path_neg)
        X,y = train.BERTEncoding()
        np.save(basicDir + 'X_0815.npy',X)
        np.save(basicDir + 'y_0815.npy',y)
        tic = time.time()
        model = train.modelTrain(X,y)
        toc = time.time()
        ## save model
        model_json = model.to_json()
        with open('./BERT_CNN_LSTM/model_selected_data_0904.json', 'w') as json:
            json.write(model_json)
        model.save_weights('./BERT_CNN_LSTM/model_selected_data_0904.h5')
        
        print(f'time collapsed: {toc-tic}')









  
