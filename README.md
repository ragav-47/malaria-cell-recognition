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
from tensorflow.keras.layers import Conv2D,MaxPool2D,AvgPool2D,Dense
from tensorflow.keras import utils
from tensorflow.keras.models import Sequential
from sklearn.metrics import classification_report,confusion_matrix
import tensorflow as tf
for device in tf.config.list_physical_devices():
    print(": {}".format(device.name))
phy_dev=tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(phy_dev))
tf.config.experimental.set_memory_growth(phy_dev[0],True)
my_data_dir = 'C:\\Users\\ragav\\Desktop\\dl\\EXP_4\\cell_images'
os.listdir(my_data_dir)
test_path = my_data_dir+'/test/'
train_path = my_data_dir+'/train/'
train_image_gen = ImageDataGenerator(rotation_range=20, # rotate the image 20 degrees
                               rescale=1./255, # Rescale the image by normalzing it.
                               zoom_range=0.1, # Zoom in by 10% max
                               horizontal_flip=True, # Allo horizontal flipping
                              )
test_image_gen = ImageDataGenerator(rotation_range=20, # rotate the image 20 degrees
                               rescale=1./255, # Rescale the image by normalzing it.
                               zoom_range=0.1, # Zoom in by 10% max
                               horizontal_flip=True, # Allo horizontal flipping
                              )
train= train_image_gen.flow_from_directory(train_path,target_size=(150,150),class_mode='binary',batch_size=16)
test=test_image_gen.flow_from_directory(test_path,class_mode = 'binary',target_size=(150,150),batch_size=16)
model=Sequential([
    Conv2D(32,3,activation="relu",padding="same"),
    AvgPool2D((2,2)),
    Conv2D(64,3,activation="relu"),
    AvgPool2D((2,2)),
    Conv2D(64,3,activation="relu"),
    AvgPool2D((2,2)),
    layers.Flatten(),
    Dense(128,activation="relu"),
    Dense(1,activation="sigmoid")
])
model.compile(loss="binary_crossentropy",optimizer="adam",metrics="accuracy")

model.fit(train,epochs=5,validation_data=test)
pd.DataFrame(model.history.history).plot()
model.evaluate(test)
pred_probabilities = model.predict(test)
predictions = pred_probabilities > 0.5
print(classification_report(test.classes,predictions))
confusion_matrix(test.classes,predictions)

import random
model=models.load_model("cell_model.h5")
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

![image](https://user-images.githubusercontent.com/75235488/196051377-e0f3caf2-fd14-482d-ba8c-c119fc0474b2.png)

### Classification Report

![image](https://user-images.githubusercontent.com/75235488/196051391-6b611227-01d0-4d89-ad42-33011a54ad97.png)

### Confusion Matrix

![image](https://user-images.githubusercontent.com/75235488/196051415-a8ba9388-50a7-4bf3-8092-1300a14d15b3.png)

### New Sample Data Prediction

![image](https://user-images.githubusercontent.com/75234991/194918608-00ba0bf5-0ea1-4d71-b015-3deea9104f16.png)

## RESULT
Thus, we have successfully created a deep neural network for Malaria infected cell recognition and analyzed the performance.
