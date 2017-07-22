
# coding: utf-8

# In[22]:

import csv
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
import sklearn
import random
from itertools import islice
samples = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in islice(reader,1,None):
        samples.append(line)
        
"""
with open('data2/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in islice(reader,1,None):
        samples.append(line)
with open('data3/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in islice(reader,1,None):
        samples.append(line)
with open('data5/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in islice(reader,1,None):
        samples.append(line)
with open('data6/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in islice(reader,1,None):
        samples.append(line)
with open('data7/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in islice(reader,1,None):
        samples.append(line)
"""

train_samples, validation_samples = train_test_split(samples, test_size=0.2, random_state=42)


# In[3]:

def generator(samples, batch_size = 32):
    num_samples = len(samples)
    while(1):
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            angles = []
            for batch_sample in batch_samples:
                index = random.randint(0,2)
                #name = batch_sample[index]
                name = 'data/IMG/'+batch_sample[index].split('/')[-1]
                image = cv2.imread(name)
                if index == 0:
                    angle = float(batch_sample[3])
                elif index == 1:
                    angle = float(batch_sample[3])+0.25
                else:
                    angle = float(batch_sample[3])-0.25
                    
                if random.randint(0,1) == 0:                
                    images.append(image)                
                    angles.append(angle)
                else:
                    images.append(cv2.flip(image,1))
                    angles.append(angle*-1.0)                          
                
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


# In[39]:

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


# In[18]:

from keras.models import *
from keras.layers import *

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20), (0,0))))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.summary()


# In[26]:

log=model.fit_generator(train_generator, samples_per_epoch= 48216, 
                    validation_data=validation_generator, 
                    nb_val_samples=len(validation_samples), nb_epoch=3)


# In[27]:

model.save('model.h5')


# In[29]:

import matplotlib.pyplot as plt
plt.plot(log.history['loss'])
plt.plot(log.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

