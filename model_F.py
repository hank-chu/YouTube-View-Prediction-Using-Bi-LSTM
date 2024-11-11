#!/usr/bin/env python
# coding: utf-8

# In[2]:

import numpy as np
import csv
import time
from sklearn.decomposition import PCA 
from sklearn.manifold import TSNE
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import tensorflow.keras.layers
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
import matplotlib.pyplot as plt


#do PCA,turn data into (len(data),1,2)
def dataPCA(data):
    docu2=data.reshape(len(data),100)
    pca = PCA(n_components=2).fit_transform(docu2)
    
    x_lab = np.array(pca)
    x_lab = x_lab.reshape(len(data),2)
    return x_lab

#do TSME,turn data into (len(data),1,2)
def dataTSNE(data):
    x_lab=[]
    docu2=data.reshape((len(data),100))
    tsne = TSNE(n_components=2, init='pca', perplexity=30).fit_transform(docu2)
    
    x_lab = np.array(tsne)
    x_lab=x_lab.reshape((len(data),1,2))
    return x_lab

#input data from file 2:sub 3:time 7:views
def readfromcsv(file,n):
    with open(file,'r' ,  encoding='UTF-8') as csvfile:
        reader = csv.reader(csvfile)
        views = [row[n] for row in reader]    
    del views[0]
    return views

#process ylabel's value, turn into integer, and divide by val (use after readfromcsv())
def process_ylabel(views,val):
    for i in range(len(views)):
        n=int(views[i])
        n=round(n/val)
        views[i] = n
    ylabel = np.array(views)
    return ylabel

#tim = readfromcsv(file,3)

#process time's value into Numerical value, tim = original time  (use after readfromcsv())
def process_time(tim):
    #use the time when the data was scraped
    timeString = "2022-08-05 20:00:00" # 時間格式為字串
    struct_time = time.strptime(timeString, "%Y-%m-%d %H:%M:%S") # 轉成時間元組
    time_stop = int(time.mktime(struct_time)) # 轉成時間戳
    
    timelist=[]
    for i in tim:
        struct_time = time.strptime(i, "%Y/%m/%d %H:%M")
        time_stamp = int(time.mktime(struct_time))
        timetonow = time_stop - time_stamp
        timelist.append(timetonow)
        
#standardization, and reshape into 3 dimension
    timedata = preprocessing.scale(timelist)
    timedata=np.array(timedata)
    
    timedata=timedata.reshape((len(mixed_data),1))
    return timedata    

#join 2 different np.array, 3dimension:n=2, 2dimension:n=1
def xlabel_join(docu2,timedata,n):
    timedata=np.array(timedata)
    x_label=np.append(docu2, timedata, axis = n )
    return x_label

#views = readfromcsv(file,2)

#process_ylabel(views,100000)

#join 2 different 2dimension np.array 
def process_sub(x_label,subs):
    subarray=np.array(subs)
    subdata=subarray.reshape((len(subarray),1))
    x_label=np.append(x_label, subdata, axis = 1 )
    return x_label

#model input x_label, ylabel, n_features ex:( 1, 4),( 1, 102)
def LSTMmodel(n_features):
    model = Sequential()
    model.add(Dense(32, input_dim=10,input_shape=n_features))
    model.add(Bidirectional(LSTM(50, activation='relu')))
    model.add(Dropout(0.5))
    model.add(Dense(units = 1 ,activation="relu"))
    model.compile(loss='mse', optimizer="adam")
    return model

#train model
#history = model.fit(x_train, y_train, epochs=10, batch_size=64 ,validation_data=((x_test, y_test)))

#save model
model.save('LSTM_model.h5')

#show the first 100 train/test result
def show_pred(model,x_train,y_train):
    pred = model.predict(x_train) #訓練好model使用predict預測看看在訓練的model跑的回歸線

    plt.plot(y_train[:100],'red' ,label='views')#畫出回歸線
    plt.plot(pred[:100],label='Predicted views')
    #plt.plot(x_train, y_train, 'o') #畫出原本的點
    plt.xlabel('Time')
    plt.ylabel('Views')
    plt.legend()
    plt.show()


# In[28]:


docu2=docu_array.reshape((len(mixed_data),100))

file=filename
#data do TSNE
x_lab =dataTSNE(docu2)

#get ylabel
views = read_fromcsv(file,7)
ylabel = process_ylabel(views,100000)

#get time label
tim = read_fromcsv(file,3)
timedata = process_time(tim)
#join time data into xlabel and dimention =2 so n = 1
x_label = xlabel_join(x_lab,timedata,1)

#get sub label 
subs = read_fromcsv(file,2)
subdata = process_ylabel(subs,100000)
#join time data into xlabel and dimention =2 
x_label = process_sub(x_label,subdata)

x_label = x_label.reshape(len(x_label),1,4)

#split into x,y train,test
(x_train, x_test, y_train, y_test) = train_test_split(x_label, ylabel, test_size=0.2)
#generate model, input_shape=(1,4)
model = LSTMmodel((1,4))

#train model
history = model.fit(x_train, y_train, epochs=10, batch_size=64 ,validation_data=((x_test, y_test)))

show_pred(model,x_train,y_train)

