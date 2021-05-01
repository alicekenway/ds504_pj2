import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

from model import Model1
import numpy as np
import pandas as pd
import random
import glob
from preprocessing import data_preprocessing_train
from preprocessing import agregate_csv
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from preprocessing import data_preprocessing_eval
def training(train,test,vail):


    

    train_X,train_Y = train
    test_X,test_Y = test
    fvail_X,fvail_Y = vail

   
  
    model = Model1()

    model.build((64,200,7))

    model.summary()

    Checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'model1', monitor='val_accuracy', verbose=1, save_best_only=True,
    save_weights_only=False, mode='max', save_freq='epoch',
    options=None
    )

    EarlyStopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', min_delta=0, patience=15, verbose=1,
        mode='max',  restore_best_weights=True
    )

    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
    optimizer=opt,
    loss='categorical_crossentropy',
    metrics=['accuracy'],
    )
    print(train_X.shape)

    model.fit(
    train_X,
    to_categorical(train_Y,num_classes=5),
    epochs=35,
    batch_size=64,
    validation_data=(test_X,to_categorical(test_Y,num_classes=5)),
    callbacks = [Checkpoint,EarlyStopping]          
    )

    
    model.load_weights('model1')
    model.save_weights('lstm_2.h5')
    model.evaluate(fvail_X, to_categorical(fvail_Y),verbose=1)
    return model

def main():
    train_frame, eval_frame = agregate_csv("data_5drivers/*.csv")

    train_frame.to_csv('train_set.csv',index=False)
    eval_frame.to_csv('eval_set.csv',index=False)
    train,test,vail = data_preprocessing_train(train_frame)

    model = training(train,test,vail)
    evals(eval_frame,model)


def evals(frame = None,model=None):

    if model is None:
        model = Model1()
        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        )

        model.load_weights('lstm_2.h5')
    if frame is None:
        frame = pd.read_csv('eval_set.csv',index_col=None, header=0)
    test_list = data_preprocessing_eval(frame)

    count = 0
    for eval_X, eval_Y in test_list:
       
        #print(eval_X.shape)
        #print(to_categorical(eval_Y,num_classes=5)[0])
        #model.evaluate(eval_X, to_categorical(eval_Y,num_classes=5),verbose=1)
        print('==================================')
        print(eval_Y[0])
       
        pred = np.argmax(model.predict(eval_X),axis=1)
        pred = np.argmax(np.bincount(pred))
        print(pred)
        if eval_Y[0] == pred:
            count+=1
    print('number of sequnece: {}'.format(len(test_list)))
    print('Right prediction {}'.format(count))
    print('accuracy: {}'.format(round(count/len(test_list),2)))

evals()