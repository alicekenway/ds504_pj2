import numpy as np
import pandas as pd
import pickle


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf
from model import Model1

def process_data(traj):
    
    frame = pd.DataFrame(traj, columns =['longitude', 'latitude','time','status']) 
    frame['time'] = pd.to_datetime(frame['time'])
    

    f1 = frame['longitude']<180 

    frame1 = frame[f1]
    f2 = frame1['latitude']<180
    frame2 = frame1[f2]
    frame = frame2
    del frame1
    del frame2

    frame['second'] = frame['time'].dt.hour * 3600 + \
             frame['time'].dt.minute * 60 + \
             frame['time'].dt.second

    frame['day'] = frame['time'].dt.day
    frame['day_sin'] = np.sin(2 * np.pi * (frame['day']-1)/30)
    frame['day_cos'] = np.cos(2 * np.pi * (frame['day']-1)/30)
    frame['second_sin'] = np.sin(2 * np.pi * (frame['second']-1)/86399)
    frame['second_cos'] = np.cos(2 * np.pi * (frame['second']-1)/86399)
    frame = frame.loc[:,~frame.columns.isin(['second','day'])]
    frame['longitude'] = frame['longitude']/180
    frame['latitude'] = frame['latitude']/180    
    #print(frame.head())
    partition = frame.sort_values(['time']).drop(['time'],axis=1).to_numpy()

    window_size=100
    full_list = []
    length = partition.shape[0]
    if length<window_size:
        last = np.expand_dims(partition[-1],axis=0)
        last = np.repeat(last,window_size-length,axis=0)
        partition = np.vstack((partition,last))
        length = window_size
    
    for i in range(0,length-window_size+1,10):
        full_list.append(partition[i:i+window_size,:])    

    full_list = np.array(full_list,dtype=np.float)
    return full_list





def run(data,model=None):

    if model is None:
        model = Model1()
        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        )

        model.load_weights('lstm_2.h5').expect_partial()
    


       
        #print(eval_X.shape)
        #print(to_categorical(eval_Y,num_classes=5)[0])
        #model.evaluate(eval_X, to_categorical(eval_Y,num_classes=5),verbose=1)
   
    pred = np.argmax(model.predict(data),axis=1)
    pred = np.argmax(np.bincount(pred))
    
    
    return pred





def run_without_input_model(datalist):
    
    model = Model1()
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
    optimizer=opt,
    loss='categorical_crossentropy',
    metrics=['accuracy'],
    )

    model.load_weights('lstm_2.h5')
    
    for traj in datalist:

        data = process_data(traj)
        run(data,model)

if __name__ == "__main__":
    with open('test.pkl','rb') as t:

        datalist = pickle.load(t)

    run_without_input_model(datalist)