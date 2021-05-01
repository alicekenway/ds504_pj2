import numpy as np
import pandas as pd
import random
import glob

def agregate_csv(route):

    all_files = glob.glob(route)
    test_ind = random.sample([i for i in range(len(all_files))],5)
    
    train = []
    test = []

    for i,filename in enumerate(all_files):

        df = pd.read_csv(filename, index_col=None, header=0)
        df['seqs'] = i
        if i in test_ind:
            test.append(df)
        else:
            train.append(df)

    train_frame = pd.concat(train, axis=0, ignore_index=True)
    test_frame = pd.concat(test, axis=0, ignore_index=True)

    return train_frame, test_frame


def data_preprocessing_train(frame):

    frame['time'] = pd.to_datetime(frame['time'])
    frame = frame.sort_values(['seqs','plate','time'])

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
    
    window_size = 100
    full_list = []
    #partitions = [frame[frame['plate']==plate].sort_values(['time']).drop(['time'],axis=1).to_numpy() for plate in [0,1,2,3,4]]
    partitions = []
    for seq in frame.seqs.unique():
        for plate in [0,1,2,3,4]:
            if not frame.loc[(frame['plate']==plate)&(frame['seqs']==seq)].empty:
                partitions.append((plate,frame.loc[(frame['plate']==plate)&(frame['seqs']==seq)].sort_values(['time']).drop(['time','seqs'],axis=1).to_numpy()))
    
    for plate,partition in partitions:
        length = partition.shape[0]
        if length<window_size:
            last = np.expand_dims(partition[-1],axis=0)
            last = np.repeat(last,window_size-length,axis=0)
            partition = np.vstack((partition,last))
            length = window_size
        for i in range(0,length-window_size+1,10):
            full_list.append((partition[i:i+window_size,1:],plate))
    
    random.seed(123)
    random.shuffle(full_list)

    train_idx = int(len(full_list)*0.8)
    fvail_idx = int(len(full_list)*0.9)
    train_X = []
    train_Y = []
    test_X = []
    test_Y = []
    fvail_X = []
    fvail_Y = []
    for array, label in full_list[0:train_idx]:
        train_X.append(array)
        train_Y.append(label)

    for array, label in full_list[train_idx:fvail_idx]:
        test_X.append(array)
        test_Y.append(label)  

    for array, label in full_list[fvail_idx:]:
        fvail_X.append(array)
        fvail_Y.append(label)  

    train_X = np.array(train_X,dtype=np.float)
    test_X = np.array(test_X,dtype=np.float)
    fvail_X = np.array(fvail_X,dtype=np.float)

    return (train_X,train_Y), (test_X,test_Y), (fvail_X,fvail_Y)


def data_preprocessing_eval(frame):

    frame['time'] = pd.to_datetime(frame['time'])
  

    frame = frame.sort_values(['seqs','plate','time'])

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
    window_size = 100
    seqs_list = []
    partitions = []
    for seq in frame.seqs.unique():
        for plate in [0,1,2,3,4]:
            if not frame.loc[(frame['plate']==plate)&(frame['seqs']==seq)].empty:
                partitions.append((plate,frame.loc[(frame['plate']==plate)&(frame['seqs']==seq)].sort_values(['time']).drop(['time','seqs'],axis=1).to_numpy()))
    
    #print(partitions[0][1].shape)
    for plate,partition in partitions:
        length = partition.shape[0]
        if length<window_size:
            last = np.expand_dims(partition[-1],axis=0)
            print(last.shape)
            last = np.repeat(last,window_size-length,axis=0)
            print(last.shape)
            partition = np.vstack((partition,last))
            length = window_size

        temp = []
        for i in range(0,length-window_size+1,10):
            temp.append((partition[i:i+window_size,1:],plate))
        
        seqs_list.append(temp)

    print(seqs_list[0][0][0][0])
    result = []
    for item in seqs_list:
        eval_X = []
        eval_Y = []
        for array, label in item:
            eval_X.append(array)
            eval_Y.append(label)

        eval_X = np.array(eval_X,dtype=np.float)
        result.append((eval_X,eval_Y))

    return result
