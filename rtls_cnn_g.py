
#!/usr/bin/env python
# coding: utf-8


import imageio
import imageio.core.util
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import re
import sklearn
import time


import keras
keras.__version__
from keras import layers
from keras import models
from keras import optimizers
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout

####################################################  Data currently in Excel files.
####################################################  Create directories for new .PNG images
####################################################


def mkNewDirs(val_dir, train_dir, SubdirsFrom):
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)

    list_classes = os.listdir(SubdirsFrom)

    for class_name in list_classes:
        os.makedirs("{}/{}".format(val_dir, class_name), exist_ok=True)
        os.makedirs("{}/{}".format(train_dir, class_name), exist_ok=True)
    return list_classes
	
list_classes  =  mkNewDirs("validation_dir", "train_dir", "ExcelData")
list_classes  =  mkNewDirs("validation_PerPerson_dir", "train_PerPerson_dir", "ExcelData")


####################################################
#################################################### Normalise the RSSI values
####################################################


### Upload one example excel file into Pandas
current_file  =  pd.read_csv("Ex 1.1_0EC3_151500.csv")


### Function: fix the coordinates.
### round the left x-value Coordinate.
### add 9 to the y-value Coordinate, then round the y-value.
def round_coordinates(current_file):
 current_file['x_round']  =  current_file['x'].round() - 1
 current_file['y_round']  =  current_file['y'] + 8
 current_file['y_round']  =  current_file['y_round'].round()
 return current_file


def normalise_rssi(current_file):

 ### create new column -- "rssi_normalised"
 current_file['rssi_normalised'] = "NaN"

 ### specify min and max "avg_rssi" values --- note: need the () at the end to ensure it is a scalar (or will give Series)
 temp_min = current_file['avg_rssi'].min()
 temp_max = current_file['avg_rssi'].max()
 
 ### assert values are within sensible range.
 assert temp_min > -200
 assert temp_max < 0
 
 #########  GET RID OF EXTREME RSSI VALUES  #########
 ### replace "avg_rssi" as Zero, if it's under -100    

 current_file.loc[current_file['avg_rssi'] < -100, 'avg_rssi'] = 0

 ### replace "rssi_normalised" as Zero, if avg_rssi is under -100
 # current_file['rssi_normalised'].where(current_file["avg_rssi"] > 100, 0)
 current_file.loc[current_file['avg_rssi'] < -100, 'rssi_normalised'] = 0

 #########  NORMALISATION OF RSSI  #########


 ### replace "rssi_normalised" as Zero, if "avg_rssi" equals the minimum "avg_rssi"
 current_file.loc[current_file['avg_rssi'] == temp_min, 'rssi_normalised'] = 0

 ### normalise "rssi_normalised", if "avg_rssi" exceeds the minimum rssi
 current_file.loc[(current_file['avg_rssi'] > temp_min) & (current_file['avg_rssi'] < 0), 'rssi_normalised'] = (current_file['avg_rssi'] - temp_min)  /  (temp_max - temp_min)
 
 return current_file


round_coordinates(current_file)
normalise_rssi(current_file)

print(current_file)


####################################################
#################################################### Build func to convert to .PNG image.
####################################################


#### Cancel the warning for loss of precision for PNG image saving.
def silence_imageio_warning(*args, **kwargs):
    pass
imageio.core.util._precision_warn = silence_imageio_warning


class_counter = 0
file_counter = 100
dir1 = "train_dir"


def convert_to_png(current_file, split_method, current_file_name):
    
    print("current_file_name: {}".format(current_file_name))
    ### create empty matrix - 34x15
    png_matrix = np.zeros([34,15])
    png_matrix

    ###########  Insert the rssi_normalised values into png_matrix  ###########
    # Works basically like this:  png_matrix[x,y] = rssi_normalised

    for counter in range(20):
        png_matrix[int(current_file.at[counter,'y_round']), int(current_file.at[counter,'x_round'])] =  current_file.at[counter,'rssi_normalised']
		#.astype(np.uint8)

    a_num = np.random.rand()
        
    png_matrix
    png_matrix.ndim
    png_matrix.shape    
    
    if split_method == "per_file":
        if a_num < 0.7:
            dir1 = "train_dir"
        if a_num >= 0.7 :
            dir1 = "validation_dir"
    if split_method == "per_name":
        train = 0
        for ID_training_persons in train_seven: 
            if ID_training_persons in current_file_name:
                train = 1
        if train == 0:
            dir1 = "validation_PerPerson_dir"
        if train == 1:
            dir1 = "train_PerPerson_dir"

    imageio.imwrite('{}/{}/npimage{}.png'.format(dir1, list_classes[class_counter], file_counter), png_matrix)

    
# test run:
# convert_to_png(current_file) 





####################################################
####################################################  Cycle through all CSVs, and apply func to save as PNG files
####################################################


##### different person IDs in each class -- so better to automate instead of hard coding each class separately.
def train_test_split_by_person(list_files):
	
    list_of_ten_person_ids  =  []

    for x in list_files:
        m = re.search('_(.+?)_', x)
        list_of_ten_person_ids.append(m.group(1))
    list_of_ten_person_ids  =  set(list_of_ten_person_ids)
    print("list_of_ten_person_ids: {}".format(list_of_ten_person_ids))

    train_seven = random.sample(list_of_ten_person_ids, 7)
    val_three   = [x for x in list_of_ten_person_ids if x not in train_seven]

    print("train_seven: {}".format(train_seven))
    print("val_three: {}".format(val_three))

    return train_seven, val_three


####################################################
####################################################  Cycle through all CSVs, and apply func to save as PNG files
####################################################

"""
### counter for the classes (directories)
class_counter = 0

very_start_time = time.time()

### iterate over classes (directories).
while class_counter < len(list_classes):
    
    start_time = time.time()

    ### list all the files associated with a particular class.
    list_files = os.listdir("ExcelData/{}".format(list_classes[class_counter]))
	
	### specify the 3 people for validation // and 3 for training:
    train_seven, val_three  =  train_test_split_by_person(list_files)

    ### create counter for files in a class.
    file_counter  = 0
    
    ### iterate over files.
    while file_counter  < len(list_files):
        ### read one file as "current_file"
        current_file  =  pd.read_csv("ExcelData/{}/{}".format(list_classes[class_counter], list_files[file_counter] ))
        current_file_name  =  list_files[file_counter]
		
        round_coordinates(current_file)
        normalise_rssi(current_file)

        convert_to_png(current_file, "per_file", current_file_name) 
        convert_to_png(current_file, "per_name", current_file_name) 

        file_counter  += 1
    class_counter += 1
    print("Class: {}, Number of files converted to PNG: {}".format(class_counter, file_counter))
    end_time = time.time()
    print("total time taken to create PNGs this class: ", end_time - start_time)
 
print("")
very_end_time = time.time()
print("total time taken to convert all classes to PNG: ", very_end_time - very_start_time)


"""

####################################################
#################################################### Prepare image files for Keras - ImageDataGenerator - Chollet book approach.
####################################################

# No need to rescale images by 1./255 --- already rescaled in RSSI section.

### per file train-test split (i.e. random number generator for each file - if <0.7 then Training data)
train_datagen = ImageDataGenerator()
validation_datagen = ImageDataGenerator()

### per person train-test split: (i.e. randomly select 7 of 10 people for training data in each class)
train_PP_datagen = ImageDataGenerator()
validation_PP_datagen = ImageDataGenerator()

### golden master -- png images --  train-test directories:
train_GM_datagen = ImageDataGenerator()
validation_GM_datagen = ImageDataGenerator()

### per file train-test split
train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        "train_dir",
        # All images will be resized to 34x15
        target_size=(34,15),
        batch_size=44,
        #  > 2 labels. (i.e. loss function is categorical_crossentropy)
        class_mode='categorical',
		color_mode= 'grayscale')

validation_generator = validation_datagen.flow_from_directory(
        "validation_dir",
        target_size=(34,15),
        batch_size=44,
        class_mode='categorical',
		color_mode= 'grayscale')
	
### per person train-test split
train_PP_generator = train_PP_datagen.flow_from_directory(
        # This is the target directory
        "train_PerPerson_dir",
        # All images will be resized to 34x15
        target_size=(34,15),
        batch_size=44,
        #  > 2 labels. (i.e. loss function is categorical_crossentropy)
        class_mode='categorical',
		color_mode= 'grayscale')	

validation_PP_generator = validation_PP_datagen.flow_from_directory(
        "validation_PerPerson_dir",
        target_size=(34,15),
        batch_size=44,
        class_mode='categorical',
		color_mode= 'grayscale')		


### GOLDEN MASTER train-test split
train_GM_generator = train_GM_datagen.flow_from_directory(
        # This is the target directory
        "GOLDEN_MASTER png data/pixeldata2/train",
        # All images will be resized to 34x15
        target_size=(34,15),
        batch_size=44,
        #  > 2 labels. (i.e. loss function is categorical_crossentropy)
        class_mode='categorical',
		color_mode= 'grayscale')	

validation_GM_generator = validation_GM_datagen.flow_from_directory(
        "GOLDEN_MASTER png data/pixeldata2/test",
        target_size=(34,15),
        batch_size=44,
        class_mode='categorical',
		color_mode= 'grayscale')			


		
		

####################################################
#################################################### Build model_1 architecture
####################################################


model_1 = models.Sequential()

model_1.add(layers.Conv2D(32, (7, 7), activation='relu', input_shape=(34, 15, 1)))
model_1.add(layers.MaxPooling2D((2, 2)))
model_1.add(Dropout(0.3))

model_1.add(layers.Conv2D(32, (3, 3), activation='relu'))
model_1.add(layers.MaxPooling2D((2, 2)))
model_1.add(Dropout(0.3))

model_1.add(layers.Flatten())
model_1.add(layers.Dense(128, activation='relu'))
model_1.add(Dropout(0.3))
model_1.add(layers.Dense(21, activation='softmax'))

model_1.summary()


####################################################
#################################################### Build model_2 architecture
#################################################### change #1: 7x7 to (5x5 and 3x3)


model_2 = models.Sequential()

model_2.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=(34, 15, 1)))
model_2.add(layers.Conv2D(32, (3, 3), activation='relu'))
model_2.add(layers.MaxPooling2D((2, 2)))
model_2.add(Dropout(0.3))

model_2.add(layers.Conv2D(32, (3, 3), activation='relu'))
model_2.add(layers.MaxPooling2D((2, 2)))
model_2.add(Dropout(0.3))

model_2.add(layers.Flatten())
model_2.add(layers.Dense(128, activation='relu'))
model_2.add(Dropout(0.3))
model_2.add(layers.Dense(21, activation='softmax'))

model_2.summary()


####################################################
#################################################### Build model_3 architecture
#################################################### change #2: dense layer to 64 nodes.


model_3 = models.Sequential()

model_3.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=(34, 15, 1)))
model_3.add(layers.Conv2D(32, (3, 3), activation='relu'))
model_3.add(layers.MaxPooling2D((2, 2)))
model_3.add(Dropout(0.3))

model_3.add(layers.Conv2D(32, (3, 3), activation='relu'))
model_3.add(layers.MaxPooling2D((2, 2)))
model_3.add(Dropout(0.3))

model_3.add(layers.Flatten())
model_3.add(layers.Dense(64, activation='relu'))
model_3.add(Dropout(0.3))
model_3.add(layers.Dense(21, activation='softmax'))

model_3.summary()


####################################################
#################################################### Compile and run model
####################################################

model_1.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# fit method using generator.
"""
history_perFile = model_1.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=100,
      validation_data=validation_generator,
      validation_steps=50)

model_1.save('cnn_perFile.h5')

history_perPerson = model_1.fit_generator(
      train_PP_generator,
      steps_per_epoch=100,
      epochs=100,
      validation_data=validation_PP_generator,
      validation_steps=50)

model_1.save('cnn_perPerson.h5')
"""

history_goldenMaster = model_1.fit_generator(
      train_GM_generator,
      steps_per_epoch=100,
      epochs=100,
      validation_data=validation_GM_generator,
      validation_steps=50)

model_1.save('cnn_goldenMaster.h5')




####################################################
####################################################
####################################################

"""
# loading model saved locally in test_model_1.h5
model_filepath = 'cnn_image.h5'
prev_model = keras.models.load_model(model_filepath)

# making the prediction
prediction = prev_model_1.predict(train_generator)

# logging each on a separate line
for single_prediction in prediction:
    print(single_prediction)
"""



####################################################
#################################################### plot loss and accuracy over training and validation data during training
####################################################

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

plt.savefig(loss_accuracy_curve)



"""


USER IDENTIFIER:
	_X_
TIME:
	_NNNNNN (6 DIGITS)



XXX change the number for "steps_per_epoch": ie this depends on the number of samples in training / test data. 
even though it will just be an approximatino, because the number can fluctuate (ie. using >0.7) --- fine to approximate it.
can later come up with a better method to split -- ie that gives consistently same number in train/test.


XXX - add another densely connected network on top -- i.e. for time series analysis.
To take advantage of the temporal information of each tag, a separate artificial neural network (ANN) was also designed, and can be seen in Fig 4. This artificial network required thirty consecutive seconds of data from a particular tag in order to make a prediction. First, the trained CNN above would classify each individual image to yield a probability vector. The probability vectors for all thirty seconds would then be combined and used as the input layer for the ANN. This included the current time,t, as well as the twenty-nine previous seconds (t−1,t−2,...,t−29). The ANN was designed with a total of three layers. The first two layers each include 125 neural nodes activated using ReLU. The final layer is a softmax layer yielding the final probability vector, the maximum of which is the final classification.  

Hyperparameters:
rmsprop
vanilla backprop
size of network
dropout settings
"""

		
		