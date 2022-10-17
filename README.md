# Deep Neural Network for Malaria Infected Cell Recognition

## AIM

To develop a deep neural network for Malaria infected cell recognition and to analyze the performance.

## PROBLEM STATEMENT AND DATASET

We have a directory with a number of images of infected cells and uninfected cells. Using an image generator, we augment the data into multiple images. We pass the data to the model and train the model accordingly using the required number of neurons.

## NEURAL NETWORK MODEL
![image](https://user-images.githubusercontent.com/75235488/196051995-53c0fa2c-5ccc-4a44-b220-bad3fe73b758.png)

## DESIGN STEPS

### Step 1:
Define the directory for the dataset. Extract the dataset files if needed.
### Step 2:
Define the image Generator engine with the necessary parameters.
### Step 3:
Pass the directory to the image generator.
### Step 4:
Define the model with appropriate neurons.
### Step 5:
Pass the training and validation data to the model.
### Step 6:
Plot the necessary graphs.

## PROGRAM
```python
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.layers import Conv2D,MaxPool2D,AvgPool2D,Dense
from tensorflow.keras import utils
from tensorflow.keras.models import Sequential
from sklearn.metrics import classification_report,confusion_matrix
import tensorflow as tf
# to share the GPU resources for multiple sessions
from tensorflow.compat.v1.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True # dynamically grow the memory used on the GPU
config.log_device_placement = True # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
set_session(sess)

%matplotlib inline
my_data_dir = '/home/ailab/hdd/dataset/cell_images'
os.listdir(my_data_dir)
test_path = my_data_dir+'/test/'
train_path = my_data_dir+'/train/'
image_shape = (130,130,3)
image_gen = ImageDataGenerator(rotation_range=20, # rotate the image 20 degrees
                               width_shift_range=0.10, # Shift the pic width by a max of 5%
                               height_shift_range=0.10, # Shift the pic height by a max of 5%
                               rescale=1/255, # Rescale the image by normalzing it.
                               shear_range=0.1, # Shear means cutting away part of the image (max 10%)
                               zoom_range=0.1, # Zoom in by 10% max
                               horizontal_flip=True, # Allo horizontal flipping
                               fill_mode='nearest' # Fill in missing pixels with the nearest filled value
                              )
image_gen.flow_from_directory(train_path)
image_gen.flow_from_directory(test_path)
model = models.Sequential()
model.add(layers.Input(shape=(130,130,3))) 
model.add(layers.Conv2D(filters=32,kernel_size=(3,3),padding="same",activation='relu'))
model.add(layers.AvgPool2D(pool_size=(2,2)))
model.add(layers.Conv2D(filters=32,kernel_size=(3,3),padding="same",activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(32,activation='relu')) 
 
model.add(layers.Dense(1, activation ='sigmoid'))
model.summary()
model.compile(loss="binary_crossentropy",optimizer="adam",metrics="accuracy")

batch_size = 16
train_image_gen = image_gen.flow_from_directory(train_path,
                                               target_size=image_shape[:2],
                                                color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='binary')
test_image_gen = image_gen.flow_from_directory(test_path,
                                               target_size=image_shape[:2],
                                                color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='binary',shuffle = False)
model.fit(train_image_gen,epochs=5,validation_data=test_image_gen)
pd.DataFrame(model.history.history).plot()
model.evaluate(test_image_gen)
pred_probabilities = model.predict(test_image_gen)
predictions = pred_probabilities > 0.5
print(classification_report(test_image_gen.classes,predictions))
confusion_matrix(test_image_gen.classes,predictions)
model.save("cell_model.h5")
import random
list_dir=["uninfected","parasitized"]
dir_=(random.choice(list_dir))
para_img= imread(train_path+
                 '/'+dir_+'/'+
                 os.listdir(train_path+'/'+dir_)[random.randint(0,10000)])
img  = tf.convert_to_tensor(np.asarray(para_img))
img = tf.image.resize(img,(130,130))
img=img.numpy()
pred=bool(model.predict(img.reshape(1,130,130,3))<0.5 )
plt.title("Model prediction: "+("Parasitized" if pred  else "Uninfected")+"\nActual Value: "+str(dir_))
plt.axis("off")
plt.imshow(img)
plt.show()
```
## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![image](https://user-images.githubusercontent.com/75235488/196158807-e6de2f9d-7e9b-41a2-9831-da02c261ee45.png)

### Classification Report

![image](https://user-images.githubusercontent.com/75235488/196158657-bfdb247a-9ae8-4982-9df4-06335538e7b4.png)

### Confusion Matrix

![image](https://user-images.githubusercontent.com/75235488/196158723-d5aa511c-a0be-454c-81fb-092794b74cb8.png)

### New Sample Data Prediction

![image](https://user-images.githubusercontent.com/75234991/194918608-00ba0bf5-0ea1-4d71-b015-3deea9104f16.png)

## RESULT
Thus, we have successfully created a deep neural network for Malaria infected cell recognition and analyzed the performance.
