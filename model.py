
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf
class Model(tf.keras.Model):
    def __init__(self):
        super(Model,self).__init__()
        
        self.Input = tf.keras.Input(shape=(200,7))
        self.lstm = tf.compat.v1.keras.layers.CuDNNLSTM(200)
        self.dense1 = tf.keras.layers.Dense(10, activation=tf.nn.relu)
        self.dense = tf.keras.layers.Dense(5, activation=tf.nn.softmax)
        self.Output = self.call(self.Input)
    def call(self,x,train=True):
        x = self.dense1(self.lstm(x))
        return self.dense(x)



import tensorflow as tf
class Model1(tf.keras.Model):
    def __init__(self):
        super(Model1,self).__init__()
        self.Input = tf.keras.Input(shape=(100,7))
        self.lstm = tf.compat.v1.keras.layers.CuDNNLSTM(200,return_sequences=True)
        self.lstm1 = tf.compat.v1.keras.layers.CuDNNLSTM(200)
        self.dense1 = tf.keras.layers.Dense(10, activation=tf.nn.sigmoid)
        self.dense = tf.keras.layers.Dense(5, activation=tf.nn.softmax)
        self.Output = self.call(self.Input)
    def call(self,x,train=True):
        x = self.lstm(x)
        x = self.lstm1(x)
        x = self.dense1(x)
        return self.dense(x)