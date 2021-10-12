#!/usr/bin/env python
# coding: utf-8

# # Cement vs Steel: Geo-assets detection strategies
# 
# ### Feature embeddings
# - Original images are labeled cement.xxx.png and steel.xxx.png

# In[12]:


#Rename image chips for training/test division

import os
import shutil, sys

def main():
    
    for count, filename in enumerate(os.listdir('/home/smit0174/Downloads/steel/')):
        dst="steel."+str(count)+".png"
        src='/home/smit0174/Downloads/steel/'+filename
        dst='/home/smit0174/Downloads/steel/'+dst
        
        os.rename(src,dst)
        
main()

def main():
    
    for count, filename in enumerate(os.listdir('/home/smit0174/Downloads/cement/')):
        dst="cement."+str(count)+".png"
        src='/home/smit0174/Downloads/cement/'+filename
        dst='/home/smit0174/Downloads/cement/'+dst
        
        os.rename(src,dst)
        
main()

#Put them into the single target directory

RootDir1 = r'/home/smit0174/Downloads/'
TargetFolder = r'/home/smit0174/DeepLearningCV2/10. Data Augmentation/datasets/images/'

for root, dirs, files in os.walk((os.path.normpath(RootDir1)), topdown=False):
    for name in files:
        if name.endswith('.png'):
            SourceFolder = os.path.join(root,name)
            shutil.copy2(SourceFolder, TargetFolder)


# In[13]:


# Get filenames in list

from os import listdir
from os.path import isfile, join

mypath = "./datasets/images/"

file_names = [f for f in listdir(mypath) if isfile(join(mypath, f))]

print(str(len(file_names)) + ' images loaded')


# ### Splitting loaded images into a training and test/validation dataset
# - need to store their labels (i.e. y_train and y_test)
# - re-size (if required!) to maintain a constant dimension
# - going to use 400 images of cement and 400 images of steel assets as training data
# - test/validation dataset uses 98 images of each class
# - cement will have labels 1 and steel 0
# - directories
#  - /datasets/cementvssteel/train/cement
#  - /datasets/cementvssteel/train/steel
#  - /datasets/cementvssteel/validation/cement
#  - /datasets/cementvssteel/validation/steel

# In[15]:


import cv2
import numpy as np
import sys
import os
import shutil

# Extract 400 for our training data and 98 for each validation sets

cement_count = 0
steel_count = 0
training_size = 400
test_size = 98
training_images = []
training_labels = []
test_images = []
test_labels = []
size = 256
cement_dir_train = "./datasets/cementvssteel/train/cement/"
steel_dir_train = "./datasets/cementvssteel/train/steel/"
cement_dir_val = "./datasets/cementvssteel/validation/cement/"
steel_dir_val = "./datasets/cementvssteel/validation/steel/"

def make_dir(directory):
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory)

make_dir(cement_dir_train)
make_dir(steel_dir_train)
make_dir(cement_dir_val)
make_dir(steel_dir_val)

def getZeros(number):
    if(number > 10 and number < 100):
        return "0"
    if(number < 10):
        return "00"
    else:
        return ""

for i, file in enumerate(file_names):
    
    if file_names[i][0] == "c":
        cement_count += 1
        image = cv2.imread(mypath+file)
        image = cv2.resize(image, (size, size), interpolation = cv2.INTER_AREA)
        if cement_count <= training_size:
            training_images.append(image)
            training_labels.append(1)
            zeros = getZeros(cement_count)
            cv2.imwrite(cement_dir_train + "cement" + str(zeros) + str(cement_count) + ".png", image)
        if cement_count > training_size and cement_count <= training_size+test_size:
            test_images.append(image)
            test_labels.append(1)
            zeros = getZeros(cement_count-400)
            cv2.imwrite(cement_dir_val + "cement" + str(zeros) + str(cement_count-400) + ".png", image)
            
    if file_names[i][0] == "s":
        steel_count += 1
        image = cv2.imread(mypath+file)
        image = cv2.resize(image, (size, size), interpolation = cv2.INTER_AREA)
        if steel_count <= training_size:
            training_images.append(image)
            training_labels.append(0)
            zeros = getZeros(steel_count)
            cv2.imwrite(steel_dir_train + "steel" + str(zeros) + str(steel_count) + ".png", image)
        if steel_count > training_size and steel_count <= training_size+test_size:
            test_images.append(image)
            test_labels.append(0)
            zeros = getZeros(steel_count-400)
            cv2.imwrite(steel_dir_val + "steel" + str(zeros) + str(steel_count-400) + ".png", image)

    if cement_count == training_size+test_size and steel_count == training_size+test_size:
        break

print("Training and Test Data Extraction Complete")


# In[16]:


# Using numpy's savez function to store our loaded data as NPZ files

np.savez('steel_vs_cement_training_data.npz', np.array(training_images))
np.savez('steel_vs_cement_training_labels.npz', np.array(training_labels))
np.savez('steel_vs_cement_test_data.npz', np.array(test_images))
np.savez('steel_vs_cement_test_labels.npz', np.array(test_labels))


# In[17]:


# Loader Function

import numpy as np

def load_data_training_and_test(datasetname):
    
    npzfile = np.load(datasetname + "_training_data.npz")
    train = npzfile['arr_0']
    
    npzfile = np.load(datasetname + "_training_labels.npz")
    train_labels = npzfile['arr_0']
    
    npzfile = np.load(datasetname + "_test_data.npz")
    test = npzfile['arr_0']
    
    npzfile = np.load(datasetname + "_test_labels.npz")
    test_labels = npzfile['arr_0']

    return (train, train_labels), (test, test_labels)


# In[55]:


(x_train, y_train), (x_test, y_test) = load_data_training_and_test("steel_vs_cement")

# Reshaping our label data from (800,) to (800,1) and test data from (196,) to (196,1)
y_train = y_train.reshape(y_train.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)

# Change our image type to float32 data type
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalize our data by changing the range from (0 to 255) to (0 to 1)
x_train /= 255
x_test /= 255

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img1 = mpimg.imread('./datasets/cementvssteel/train/cement/cement002.png')
img2 = mpimg.imread('./datasets/cementvssteel/train/cement/cement006.png')
img3 = mpimg.imread('./datasets/cementvssteel/train/cement/cement007.png')
img4 = mpimg.imread('./datasets/cementvssteel/train/steel/steel002.png')
img5 = mpimg.imread('./datasets/cementvssteel/train/steel/steel006.png')
img6 = mpimg.imread('./datasets/cementvssteel/train/steel/steel007.png')


fig = plt.figure(figsize=(10, 10))

ax1 = fig.add_subplot(2,2,1)
ax1.imshow(img1)
ax1.set_title('CEMENT')

ax2 = fig.add_subplot(2,2,2)
ax2.imshow(img4)
ax2.set_title('STEEL')

ax3 = fig.add_subplot(2,2,3)
ax3.imshow(img2)

ax4 = fig.add_subplot(2,2,4)
ax4.imshow(img5)


# In[57]:


#**Sigmoids are used when we're doing binary classification
#Note the binary_crossentropy loss

from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os

batch_size = 16
epochs = 25

img_rows = x_train[0].shape[0]
img_cols = x_train[1].shape[0]
input_shape = (img_rows, img_cols, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

print(model.summary())


# In[58]:


#Model training

history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True)

# Evaluate the performance of our trained model
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])


# In[59]:


# Plotting our loss charts
import matplotlib.pyplot as plt

history_dict = history.history

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)

line1 = plt.plot(epochs, val_loss_values, label='Validation Loss')
line2 = plt.plot(epochs, loss_values, label='Training Loss')
plt.setp(line1, linewidth=2.0, marker = '+', markersize=10.0)
plt.setp(line2, linewidth=2.0, marker = '4', markersize=10.0)
plt.xlabel('Epochs') 
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.show()


# In[60]:


# Plotting our accuracy charts
import matplotlib.pyplot as plt

history_dict = history.history

acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
epochs = range(1, len(loss_values) + 1)

line1 = plt.plot(epochs, val_acc_values, label='Validation Accuracy')
line2 = plt.plot(epochs, acc_values, label='Training Accuracy')
plt.setp(line1, linewidth=2.0, marker = '+', markersize=10.0)
plt.setp(line2, linewidth=2.0, marker = '4', markersize=10.0)
plt.xlabel('Epochs') 
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()
plt.show()


# In[61]:


from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

y_pred = model.predict_classes(x_test)

print(classification_report(np.argmax(y_test,axis=1), y_pred))
print(confusion_matrix(np.argmax(y_test,axis=1), y_pred))

# TP | FP
# -------
# FN | TN

#Cement=1, steel =0


# In[26]:


model.save(r'/home/smit0174/DeepLearningCV2/Trained Models/steel_vs_cement_V1.h5')

#import cv2
#import numpy as np
#from keras.models import load_model

#classifier = load_model(r'/home/smit0174/DeepLearningCV2/Trained Models/steel_vs_cement_V1.h5')

#HOW EACH MODEL PERFORMS ON UNSEEN DATA (CHINA)?


# # Data augmentation

# In[62]:


(x_train, y_train), (x_test, y_test) = load_data_training_and_test("steel_vs_cement")

# Reshaping our label data from (800,) to (800,1) and test data from (196,) to (196,1)
y_train = y_train.reshape(y_train.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)

# Change our image type to float32 data type
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalize our data by changing the range from (0 to 255) to (0 to 1)
x_train /= 255
x_test /= 255

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[63]:


#See StackOverflow question: Why is validation accuracy higher than training accuracy when applying data augmentation?

#Answer: no augmentation is applied to test data

import os
import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras import optimizers
import scipy
import pylab as pl
import matplotlib.cm as cm
get_ipython().run_line_magic('matplotlib', 'inline')

input_shape = (256, 256, 3)
img_width = 256
img_height = 256

nb_train_samples = 800
nb_validation_samples = 196
batch_size = 16
epochs = 50

train_data_dir = './datasets/cementvssteel/train'
validation_data_dir = './datasets/cementvssteel/validation'

# Creating our data generator for our test data
validation_datagen = ImageDataGenerator(
    # used to rescale the pixel values from [0, 255] to [0, 1] interval
    rescale = 1./255)

# Creating our data generator for our training data
train_datagen = ImageDataGenerator(
      rescale = 1./255,              # normalize pixel values to [0,1]
      rotation_range = 30,           # randomly applies rotations
      width_shift_range = 0.3,       # randomly applies width shifting
      height_shift_range = 0.3,      # randomly applies height shifting
      horizontal_flip = True,        # randonly flips the image
      fill_mode = 'nearest')         # uses the fill mode nearest to fill gaps created by the above

# Specify criteria about our training data, such as the directory, image size, batch size and type 
# automagically retrieve images and their classes for train and validation sets
train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size = (img_width, img_height),
        batch_size = batch_size,
        class_mode = 'binary',
        shuffle = True)

validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size = (img_width, img_height),
        batch_size = batch_size,
        class_mode = 'binary',
        shuffle = False)    


# In[64]:


# Constructing model

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

print(model.summary())

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


# In[65]:


history = model.fit_generator(
    train_generator,
    steps_per_epoch = nb_train_samples // batch_size,
    epochs = epochs,
    validation_data = validation_generator,
    validation_steps = nb_validation_samples // batch_size)


# ## Plotting loss and accuracy

# In[66]:


# Plotting our loss charts
import matplotlib.pyplot as plt

history_dict = history.history

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)

line1 = plt.plot(epochs, val_loss_values, label='Validation Loss')
line2 = plt.plot(epochs, loss_values, label='Training Loss')
plt.setp(line1, linewidth=2.0, marker = '+', markersize=10.0)
plt.setp(line2, linewidth=2.0, marker = '4', markersize=10.0)
plt.xlabel('Epochs') 
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.show()


# In[67]:


history_dict = history.history
print(history_dict)


# In[68]:


# Plotting our accuracy charts
import matplotlib.pyplot as plt

history_dict = history.history

acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
epochs = range(1, len(loss_values) + 1)

line1 = plt.plot(epochs, val_acc_values, label='Validation Accuracy')
line2 = plt.plot(epochs, acc_values, label='Training Accuracy')
plt.setp(line1, linewidth=2.0, marker = '+', markersize=10.0)
plt.setp(line2, linewidth=2.0, marker = '4', markersize=10.0)
plt.xlabel('Epochs') 
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()
plt.show()

