
#!/usr/bin/env python
# coding: utf-8


import imageio
import imageio.core.util
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
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


####################################################  Data currently in Excel files.
####################################################  Create directories for new .PNG images
####################################################


import os
os.makedirs("validation_dir", exist_ok=True)
os.makedirs("train_dir", exist_ok=True)

list_classes = os.listdir("ExcelData")

for class_name in list_classes:
    os.makedirs("validation_dir/{}".format(class_name), exist_ok=True)
    os.makedirs("train_dir/{}".format(class_name), exist_ok=True)


####################################################
#################################################### Normalise the RSSI values
####################################################




### Upload one excel file into Pandas
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
 # current_file["avg_rssi"].where(current_file["avg_rssi"] < -90, 0)

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
dir2 = "Clean"



def convert_to_png(current_file):

    ### create empty matrix - 15x30
    png_matrix = np.zeros([30,15])
    png_matrix

    ###########  Insert the rssi_normalised values into png_matrix  ###########
    # Works basically like this:  png_matrix[x,y] = rssi_normalised

    for counter in range(20):
        png_matrix[int(current_file.at[counter,'y_round']), int(current_file.at[counter,'x_round'])] =  current_file.at[counter,'rssi_normalised'].astype(np.uint8)

    a_num = np.random.rand()
        
    png_matrix
    png_matrix.ndim
    png_matrix.shape    
     
    if a_num < 0.7:
        dir1 = "train_dir"
    if a_num >= 0.7 :
        dir1 = "validation_dir"
        
    imageio.imwrite('{}/{}/npimage{}.png'.format(dir1, list_classes[class_counter], file_counter), png_matrix)

    
# test run:
# convert_to_png(current_file) 




####################################################
####################################################  Cycle through all CSVs, and apply func to save as PNG files
####################################################


### counter for the classes (directories)
class_counter = 0

very_start_time = time.time()

### iterate over classes (directories).
while class_counter < len(list_classes):
    
    start_time = time.time()

    ### list all the files associated with a particular class.
    list_files = os.listdir("ExcelData/{}".format(list_classes[class_counter]))
    ### create counter for files in a class.
    file_counter  = 0
    
    ### iterate over files.
    while file_counter  < len(list_files):
        ### read one file as "current_file"
        current_file  =  pd.read_csv("ExcelData/{}/{}".format(list_classes[class_counter], list_files[file_counter] ))
        
        round_coordinates(current_file)
        normalise_rssi(current_file)

        convert_to_png(current_file) 
        
        file_counter  += 300
    class_counter += 1
    print("Class: {}, Number of files: {}".format(class_counter, file_counter))
    end_time = time.time()
    print("total time taken this class: ", end_time - start_time)
 
print("")
very_end_time = time.time()
print("total time taken all classes: ", very_end_time - very_start_time)



####################################################
#################################################### Prepare image files for Keras - Chollet approach.
####################################################

# No need to rescale images by 1./255 --- because already rescaled in RSSI section.
train_datagen = ImageDataGenerator()
validation_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        "train_dir",
        # All images will be resized to 30x15
        target_size=(30,15),
        batch_size=20,
        #  > 2 labels. (i.e. loss function is categorical_crossentropy)
        class_mode='categorical',
		color_mode= 'grayscale')

validation_generator = validation_datagen.flow_from_directory(
        "validation_dir",
        target_size=(30,15),
        batch_size=20,
        class_mode='categorical',
		color_mode= 'grayscale')
		
		
		
####################################################
#################################################### Build model architecture
####################################################


model = models.Sequential()

model.add(layers.Conv2D(32, (7, 7), activation='relu', input_shape=(30, 15, 1)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(20, activation='softmax'))

model.summary()


####################################################
#################################################### Compile and run the model
####################################################

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# fit method when using generator.
history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=100,
      validation_data=validation_generator,
      validation_steps=50)


test_loss, test_acc = model.evaluate(test_images, test_labels)

test_acc


model.save('cnn_image.h5')

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



"""
XXX change the number for "steps_per_epoch": ie this depends on the number of samples in training / test data. 
even though it will just be an approximatino, because the number can fluctuate (ie. using >0.7) --- fine to approximate it.
can later come up with a better method to split -- ie that gives consistently same number in train/test.

XXX to do - check is there an extra folder in here (ie the pre-processed oen which includes all of the images averaged to 1Hz (obviously this would duplicate the other data files so get rid of it.

XXX - add another densely connected network on top -- i.e. for time series analysis.
To take advantage of the temporal information of each tag, a separate artificial neural network (ANN) was also designed, and can be seen in Fig 4. This artificial network required thirty consecutive seconds of data from a particular tag in order to make a prediction. First, the trained CNN above would classify each individual image to yield a probability vector. The probability vectors for all thirty seconds would then be combined and used as the input layer for the ANN. This included the current time,t, as well as the twenty-nine previous seconds (t−1,t−2,...,t−29). The ANN was designed with a total of three layers. The first two layers each include 125 neural nodes activated using ReLU. The final layer is a softmax layer yielding the final probability vector, the maximum of which is the final classification.  

# model.compile(optimizer='rmsprop',
	i.e. will try this stuff for hyperparameter optimisation.
"""
